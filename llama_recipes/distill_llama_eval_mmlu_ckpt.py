# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""
Evaluate MMLU 

torchrun --nnodes 1 --nproc_per_node 8 llama_recipes/distill_llama_eval_mmlu_ckpt.py \
--model_config llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
--distill_config llama3_1_70b/distill_rp_llama_70b_xent1_mse1000_lr1e-2 \
--finetune_config llama3_1_70b/finetune_rp_llama_70b_qkvo \
--eval_config eval_mmlu_debug \
--verbose --replicate 0 --seed 0 --lk_zero_init --enable_fsdp --low_cpu_fsdp \
--load_finetune_checkpoint ckpt_lora-dl-d=distill_rp_llama_70b_xent1_mse1000_lr1e-2-m=distill_llama3_1_70b_lk_smd_wtk64_fd64_w01-f=finetune_rp_llama_70b-dcs=1024-se=0-re=4-lzi=1-dcs=1024-se=0-re=4.pt

"""
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
    train as _train_normal,
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
# from llama_recipes.trainer_finetune_chunked import train as train_chunked

from accelerate.utils import is_xpu_available

# -------------
# Our arguments
# -------------
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
    setup_fsdp_config
)

from src.dataloaders.eval_mmlu import load_data


def check_state_dict_keys(_keys, layer_idx, rank: int = 0):
    try:
        assert len(_keys.unexpected_keys) == 0
        if rank == 0:
            print_header(f'*** All expected keys matched successfully {layer_idx} ***')
    except Exception as e:
        if rank == 0:
            print(e)
            print_header('*** Error: unexpected keys in checkpoint ***')
            print(f'Unexpected keys at {layer_idx}:')
            for k in _keys.unexpected_keys:
                print(k)


def main():
    # ---------
    # 1. SET UP
    # ---------
    args = get_args()
    args.checkpoint_dir = join(args.checkpoint_dir, args.model_config)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # dl-d=llama3_1_70b/distill_rp_llama_70b_xent1_mse1000_lr1e-2-m=llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01-f=llama3_1_70b/finetune_rp_llama_70b_qkvo-se=0-re=0-lzi=1
    args.run_name += f'-e={args.eval_config}'

    kwargs = vars(args)

    # if 'finetune_long' in args.finetune_config:
    #     train = train_chunked
    # else:
    #     train = _train_normal
    train = _train_normal

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

    try:
        if not os.path.exists(model_config.model.pretrained_model_name_or_path):
            print(f"Model path {model_config.model.pretrained_model_name_or_path} does not exist. Using backup path. {model_config.model.pretrained_model_name_or_path_backup}")
            model_config.model.pretrained_model_name_or_path = model_config.model.pretrained_model_name_or_path_backup
        model_config.model.pop("pretrained_model_name_or_path_backup")
    except:
        print(f"Model without model.pretrained_model_name_or_path_backup path")
        pass

    # Update dataset pretrained model config
    for k in distill_config.dataset.pretrained_model_config:
        distill_config.dataset.pretrained_model_config[k] = getattr(model_config.model, k, None)

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

    wandb_run = None

    if not args.no_wandb:
        if not args.enable_fsdp or rank==0:
            wandb_run = setup_wandb(distill_config, fsdp_config, **kwargs)  

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

    if 'llama' in model_config.model.pretrained_model_name_or_path:
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

    # Prepare the model for int8 training if quantization is enabled
    # -> But we only use this script for FSDP without quantization
    # if train_config.quantization:
    #     model = prepare_model_for_int8_training(model)

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
        
        # if distill_config.trainer.name is not None and args.load_distill_checkpoint is not None:
        #     if args.load_distill_checkpoint is not None:
        #         model = load_sharded_model_single_gpu(model, model_path=args.load_distill_checkpoint, cfg=distill_config, rank=rank)
        #     else:
        #         model = load_sharded_model_single_gpu(model, model_path=None, cfg=distill_config, rank=rank)
        print(" -> Proceeding without learned linear attentions")  # load these with the checkpoint

    if wandb_run and distill_peft_config is not None:
        wandb_run.config.update(distill_peft_config)
        
    # ----------------------------
    # 4. ADD FINETUNING PARAMETERS
    # ----------------------------
    finetune_config, args = prepare_finetune_configs(args, model_config, 
                                                     args.finetune_config)
    # finetune_config = update_config_from_args(finetune_config, args)
    finetune_config = setup_fsdp_config(finetune_config, args, 'finetune')
    if args.finetune_lr is not None:
        finetune_config.model_name += f'=flr={args.finetune_lr}'
            
    # model, ft_peft_config
    model, _ = load_and_convert_finetune(model, finetune_config,
                                         checkpoint_path=None,
                                         print_model=args.verbose,
                                         merge_loras=False,
                                         peft_gradient_checkpointing=not args.no_peft_grad_ckpt,
                                         rank=rank)

    if args.load_finetune_checkpoint is not None:
        if '.pt' in args.load_finetune_checkpoint:
            with torch.no_grad():
                _keys = model.load_state_dict(torch.load(args.load_finetune_checkpoint), strict=False)
                check_state_dict_keys(_keys, 0)
        else:
            model = load_sharded_model_single_gpu(model, model_path=args.load_finetune_checkpoint, cfg=finetune_config, rank=rank)
    else:
        print(" -> Proceeding without finetuned parameters")

    if rank == 0:
        print_header('** Sanity check model weights **')
        for n, p in model.named_parameters():
            if ('layers.0.' in n and ('feature_map' in n or 'lora' in n or 'weight_factors' in n)):
                print(f'-> {n}:\n', p)
    

    # ------------------------------------------------------
    # 5. SETUP FSDP AND LOAD DISTILLED ATTENTION CHECKPOINTS
    # ------------------------------------------------------
    if args.enable_fsdp:

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank, model=model_type)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, DecoderLayer)
        
        device_id = 0
        if is_xpu_available():
            device_id = torch.xpu.current_device()
        elif torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            print('-> device_id:', device_id)

        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrapping_policy,  # if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            # device_mesh=hsdp_device_mesh,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=args.low_cpu_fsdp,  # train_config.low_cpu_fsdp
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if args.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
    
    else:  # if not model_config.model.quantization and not args.enable_fsdp:
        if is_xpu_available():
            model.to("xpu:0")
        elif torch.cuda.is_available():
            model.to("cuda")

    if args.verbose and (rank == 0 or not args.enable_fsdp):
        print_header('*** FSDP MODEL ***')
        print(model)
        print_header('*** Trainable Parameters ***')
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f'├── {n} (dtype = {p.dtype})')
        # print_header('*** model.state_dict() ***')
        # for k in model.state_dict().keys():
        #     print(f'├── {k}')

    # -----------
    # 5. EVALUATE
    # -----------
    from llama_recipes.trainer_eval_mmlu import evaluate_mmlu

    # Get data
    eval_config = f'./configs/experiment/{args.eval_config}.yaml'
    eval_config = OmegaConf.load(eval_config)
    for k in eval_config.dataset.pretrained_model_config:
        eval_config.dataset.pretrained_model_config[k] = getattr(model_config.model, k, None)

    if rank == 0 or not args.enable_fsdp:
        print_header('*** Eval Config ***')
        if args.verbose:
            print_config(eval_config)
        
    eval_dataloader = load_data(**eval_config['dataset'], **eval_config['dataloader'])
    
    if not args.enable_fsdp or rank == 0:
        print(f"--> Validation Set Length = {len(eval_dataloader.dataset)}")

    results = evaluate_mmlu(
        model, 
        finetune_config, 
        eval_dataloader,
        local_rank, 
        tokenizer,
        wandb_run,
        epoch=0,
        rank=rank,
    )

    if not args.enable_fsdp or rank==0:
        for k,v in results.items():
            print(f'{k}: {v}')  
        if 'all' in results:
            print_header('*** MMLU ***')
            for k, v in results['all'].items():
                print(f'{k}: {v}')

if __name__ == "__main__":
    main()
