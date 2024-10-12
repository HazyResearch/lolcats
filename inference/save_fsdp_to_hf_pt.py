# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""
Convert model to .pt state_dict

torchrun --nnodes 1 --nproc_per_node 7 llama_recipes/save_fsdp_to_hf_pt.py \
--model_config distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_redpajama_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_redpajama \
--eval_config eval_alpaca_clean --verbose --replicate 4 --seed 0 --lk_zero_init \
--eval_steps 10 --dataset_chunk_size 512 --enable_fsdp --low_cpu_fsdp \
--load_distill_checkpoint /data/rahul/checkpoints/rp_0907/finetune_rp_llama_70b-fac=1-dcs=1024-se=0-re=0-lzi=1 \
--load_finetune_checkpoint /data/rahul/checkpoints/rp_0907/finetune_rp_llama_70b_qkvo-fac=1-dcs=1024-se=0-re=0-lzi=1-dcs=1024-se=0-re=0 

"""
from logging import StringTemplateStyle
import os
from os.path import join
# import sys
# sys.path.append('/workspace/lolcats')  # needed for vast-ai instances
import dataclasses
import random
import argparse  # ours
from pkg_resources import packaging

import torch
import torch.optim as optim

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType
)

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
# from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from llama_recipes.utils.fsdp_utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    # generate_peft_config,
    # generate_dataset_config,
    # get_dataloader_kwargs,
)
from llama_recipes.utils.fsdp_utils import (
    hsdp_device_mesh as get_hsdp_device_mesh
)
from llama_recipes.trainer_finetune import (
    train,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)
from llama_recipes.model_checkpointing.distill_checkpoint_handler import (
    load_model_sharded,
    load_sharded_model_single_gpu,
)

from accelerate.utils import is_xpu_available

# -------------
# Our arguments
# -------------
from safetensors.torch import save_file
from omegaconf import OmegaConf

from src.utils.setup import (
    update_config_from_args,
    update_model_config_from_args
)
from src.utils.logging import print_header, print_config
# from src.dataloaders import load_data
from src.trainer import get_scheduler

from src.finetune import prepare_finetune_configs  # get_finetuner

from src.model.pretrained import get_pretrained_loader
from src.model.load_model import (
    load_and_convert_attns,
    load_and_convert_finetune
)
from distill_llama import (
    setup_wandb, get_args,  # get_run_name_from_checkpoint
    get_dataloaders, setup_fsdp_config
)


def main():
    # ---------
    # 1. SET UP
    # ---------
    args = get_args()
    args.enable_fsdp = True
    args.checkpoint_dir = join(args.checkpoint_dir, args.model_config)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    kwargs = vars(args)

    # Load distillation + attention configs
    distill_config_path = join('./configs/experiment', f'{args.distill_config}.yaml')
    distill_config = OmegaConf.load(distill_config_path)
    distill_config = update_config_from_args(distill_config, args)

    model_config_path = join('./configs/model', f'{args.model_config}.yaml')
    model_config = OmegaConf.load(model_config_path)
    model_config = update_model_config_from_args(model_config, args)
    if args.enable_fsdp:
        if getattr(model_config.model, 'load_in_4bit', False):
            model_config.model.device_map = 'auto'
        elif getattr(model_config.model, 'load_in_8bit', False):
            model_config.model.device_map = 'auto'
        else:
            model_config.model.device_map = None  # FSDP will complain about device placement o.w.

    # Update dataset pretrained model config
    for k in distill_config.dataset.pretrained_model_config:
        distill_config.dataset.pretrained_model_config[k] = getattr(model_config.model, k)

    args.run_name = args.run_name.replace('True', '1').replace('False', '0')  # concise hacks
    
    # Update the configuration for the training and sharding process
    distill_config = setup_fsdp_config(distill_config, args, 'distill')  # patch llama-recipes args
    fsdp_config = FSDP_CONFIG()
    update_config((fsdp_config), **vars(args))
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0 or not args.enable_fsdp:
        print_header('Distillation Config')
        print_config(distill_config)
        print_header('Model Config')
        print_config(model_config)
        print_header('FSDP Config')
        print_config(dataclasses.asdict(fsdp_config))

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # ------------------------
    # 2. LOAD PRETRAINED MODEL
    # ------------------------

    # Load the pre-trained model and setup its configuration
    # Initialize tokenizer and model loader
    model_loader = get_pretrained_loader(**model_config.model, 
                                         huggingface_token=args.huggingface_token)
    tokenizer = model_loader.load_tokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    use_cache = False if args.enable_fsdp else None

    from transformers import LlamaConfig as ModelConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer as DecoderLayer
    from src.model.modeling_llama import LolcatsLlamaForCausalLM as ModelClass
    model_type = 'llama'

    # Convert model
    try:
        args.attention_type = model_config['attention']['attention_type']
    except AttributeError:
        args.attention_type = 'lolcats_llama'
    
    if args.enable_fsdp and args.low_cpu_fsdp:
        # for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        # this avoids cpu oom when loading large models like llama 70B, in which case
        # model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        # overhead and currently requires latest nightly.                
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly and rank == 0:
            print(f'-> Pytorch version is {v} ({v.dev})')
            print(f'   - Llama-recipes says "latest pytorch nightly build is required to run with low_cpu_fsdp config"')
            print(f"   - But who knows maybe this will work. We're just trying stuff.")
            print(f"   - (Also if PyTorch was installed after July 1, 2023 we should be good)")
        #     raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            # "please install latest nightly.")
        model = model_loader.load(args.attention_type)
        model.state_chunk_len = model_config['attention']['state_chunk_len']
    else:
        model = model_loader.load(args.attention_type)
        model.state_chunk_len = model_config['attention']['state_chunk_len']

    if rank == 0 or not args.enable_fsdp:
        print_header('Pretrained Model')
    
    model_config.model_name = model_config.model.pretrained_model_name_or_path
    print_model_size(model, model_config, rank if args.enable_fsdp else 0)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if args.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    # -------------------------------
    # 3. CONVERT DISTILLED ATTENTIONS
    # -------------------------------
    model, distill_peft_config = load_and_convert_attns(model, model_config,
                                                        attention_type=args.attention_type,
                                                        checkpoint_path=None,  # args.load_distill_checkpoint, 
                                                        print_model=args.verbose,
                                                        merge_loras=False,
                                                        peft_gradient_checkpointing=not args.no_peft_grad_ckpt,
                                                        train_attention=False,
                                                        rank=rank)
    if rank == 0:
        print_header('** Sanity check model weights **')
        for n, p in model.named_parameters():
            if ('layers.0.' in n and ('feature_map' in n or 'lora' in n)):
                print(f'-> {n}:\n', p)
        if args.load_distill_checkpoint is not None:
            model = load_sharded_model_single_gpu(model, model_path=args.load_distill_checkpoint, cfg=distill_config, rank=rank)
        else:
            model = load_sharded_model_single_gpu(model, model_path=None, cfg=distill_config, rank=rank)

    
    # ----------------------------
    # 4. ADD FINETUNING PARAMETERS
    # ----------------------------
    finetune_config, args = prepare_finetune_configs(args, model_config, 
                                                     args.finetune_config)
    # finetune_config = update_config_from_args(finetune_config, args)
    finetune_config = setup_fsdp_config(finetune_config, args, 'finetune')
            
    # model, ft_peft_config
    model, _ = load_and_convert_finetune(model, finetune_config,
                                         checkpoint_path=None,
                                         print_model=args.verbose,
                                         merge_loras=False,
                                         peft_gradient_checkpointing=not args.no_peft_grad_ckpt,
                                         rank=rank)

    # ------------------------------------------------------
    # 5. SETUP FSDP AND LOAD DISTILLED ATTENTION CHECKPOINTS
    # ------------------------------------------------------
    mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank, model=model_type)
    my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, DecoderLayer)
    
    device_id = 0
    if is_xpu_available():
        device_id = torch.xpu.current_device()
    elif torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        print('-> device_id:', device_id)

    # Load distilled checkpoints
    if args.verbose and rank == 0:
        print_header('*** FSDP Model ***')
        print(model)
        print('Loading checkpoints from:', distill_config.model_name)
        
    # load_model_sharded(model, rank, distill_config, model_path=args.load_distill_checkpoint)
    
    if rank == 0 or not args.enable_fsdp:  # debugging
        print_header('** Sanity check model weights **')
        for n, p in model.named_parameters():
            if ('layers.0.' in n and 'base_attn' not in n and 
                '.0.mlp.' not in n and '.block_sparse_moe' not in n):
                print(f'-> {n}:\n', p)

    if args.verbose and (rank == 0 or not args.enable_fsdp):
        print_header('*** FSDP MODEL ***')
        print(model)
        print_header('*** Trainable Parameters ***')
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f'├── {n} (dtype = {p.dtype})')
                
    # Load sharded weights across GPUs into model
    # ignore_param_rule = lambda n, p: not p.requires_grad
    # load_model_sharded(model, rank, finetune_config, ignore_param_rule)
    # model = load_sharded_model_single_gpu(model, model_path=None, cfg=finetune_config, rank=rank)

    if args.load_finetune_checkpoint is not None:
        model = load_sharded_model_single_gpu(model, model_path=args.load_finetune_checkpoint, cfg=finetune_config, rank=rank)
    else:
        model = load_sharded_model_single_gpu(model, model_path=None, cfg=finetune_config, rank=rank)
    
    
    if rank == 0:  # debugging
        print_header('** Sanity check model weights **')
        for n, p in model.named_parameters():
            if ('layers.0.' in n and 'base_attn' not in n and 
                '.0.mlp.' not in n and '.block_sparse_moe' not in n):
                print(f'-> {n}:\n', p)

    print(model.config)
    # model.save_pretrained(f'ckpt-lora_hf-{args.run_name}')
    if rank == 0:
        with torch.no_grad():
            state_dict = model.state_dict()
            keys_to_keep = [key for key in state_dict.keys() if 'lora' in key or 'window_factors' in key or 'feature_map' in key]
            # keys_to_keep = [key for key in state_dict.keys() if 'lora' in key]
            new_state_dict = {key: state_dict[key] for key in keys_to_keep}
            torch.save(new_state_dict, f'ckpt_lora-{args.run_name}.pt')

        print_header('*** Weights in state_dict ***')
        for k in torch.load(f'ckpt_lora-{args.run_name}.pt'):
            print(k)
        print('-> Checkpoints saved to:', f'ckpt_lora-{args.run_name}.pt')
        # save_file(new_state_dict, f"ckpt-{args.run_name}.safetensors")

if __name__ == "__main__":
    main()
