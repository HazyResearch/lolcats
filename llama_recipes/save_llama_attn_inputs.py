# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""
Finetune attention-swapped model. Rough adaptation of llama_recipes script for distillation.

Example usage (using the same args as distill_llama.py for convenience (just swap the file called)
```
torchrun --nnodes 1 --nproc_per_node 8 \
llama_recipes/save_llama_attn_inputs.py \
--model_config distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_llama3_1_70b_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean_llama3_1_70b \
--verbose --replicate 0 --seed 0 \
--enable_fsdp --low_cpu_fsdp 
```

"Another one"
```
torchrun --nnodes 1 --nproc_per_node 1 \
llama_recipes/save_llama_attn_inputs.py \
--model_config distill_llama3_8b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_xent0_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean \
--verbose --replicate 0 --seed 0 \
--enable_fsdp --low_cpu_fsdp 
```
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
import torch.distributed as dist

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

from tqdm import tqdm


CUTOFF_BATCH = 500   # Save to disk and delete tensors every CUTOFF_BATCH
                     # to save CPU memory


def main():
    # ---------
    # 1. SET UP
    # ---------
    args = get_args()
    args.checkpoint_dir = join(args.checkpoint_dir, args.model_config)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    kwargs = vars(args)

    if args.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0

    # Load distillation config
    # -> Compute layer-wise outputs with the dataset specified here
    distill_config_path = join('./configs/experiment', f'{args.distill_config}.yaml')
    distill_config = OmegaConf.load(distill_config_path)
    dataset_name = distill_config.dataset.name
    cache_dir = distill_config.dataset.dataset_config.cache_dir

    # Load model config
    model_config_path = join('./configs/model', f'{args.model_config}.yaml')
    model_config = OmegaConf.load(model_config_path)
    model_name = model_config.model.pretrained_model_name_or_path.replace('/', '_')
    
    # Create data directory where we'll store the layer-wise input tensors
    if rank == 0 or not args.enable_fsdp:
        data_dir = join(cache_dir, dataset_name) 
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        data_dir = join(data_dir, model_name)  # meta-llama_Meta-Llama-3.1-70B/attn_inputs-l=31-split=train.pt
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        print(f'-> Saving layer-wise tensors to {data_dir}')
    # dist.barrier()

    # Copied from distill_llama.py and distill_llama_finetune.py for FSPD
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
    
    # Update the configuration for the training and sharding process
    distill_config = setup_fsdp_config(distill_config, args, 'distill')  # patch llama-recipes args
    fsdp_config = FSDP_CONFIG()
    update_config((fsdp_config), **vars(args))
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

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
    model = model_loader.load('softmax')  # Load the original Transformer models

    if rank == 0 or not args.enable_fsdp:
        print_header('Pretrained Model')
    
    model_config.model_name = model_config.model.pretrained_model_name_or_path
    print_model_size(model, model_config, rank if args.enable_fsdp else 0)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if args.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # --------------
    # 5. SETUP FSDP 
    # --------------
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

    # Get data
    train_dataloader, eval_dataloader, distill_config = get_dataloaders(distill_config, tokenizer,
                                                                        no_shuffle_train = True)
    if not args.enable_fsdp or rank == 0:
        print(f"--> Training   Set Length = {len(train_dataloader.dataset)}")
        print(f"--> Validation Set Length = {len(eval_dataloader.dataset)}")

    # -----------------------------------
    # Compute dataset layer-wise outputs
    # -----------------------------------
    for split, dataloader in {'train': train_dataloader, 'validation': eval_dataloader}.items():
        if rank == 0 or not args.enable_fsdp:
            print_header(f'*** Computing layer-wise {split} outputs ***')

        attn_inputs_by_layer = [[] for _ in range(len(model.model.layers))]
        max_layer_digits = len(str(len(attn_inputs_by_layer)))
        with torch.no_grad():
            model.eval()
            pbar = tqdm(dataloader, desc=f'❯❯❯ Computing layer-wise outputs on rank {rank} for {split} split')
            max_digits = len(str(len(pbar)))
            for step, batch in enumerate(pbar):
                batch = {'input_ids': batch['input_ids']}
                key = 'input_ids'
                if distill_config.enable_fsdp:
                    if is_xpu_available():
                        batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                    else:
                        batch[key] = batch[key].to(local_rank)
                else:
                    if is_xpu_available():
                        batch[key] = batch[key].to('xpu:0')
                    else:
                        batch[key] = batch[key].to('cuda:0')
                # Save tensors -> HuggingFace Llama API
                outputs = model(**batch, output_hidden_states=True, use_cache=False).get('hidden_states')
                for idx, hidden_state in enumerate(outputs[:-1]):  # indexed by layer, last layer is hidden layer before lm_head
                    hidden_state = model.model.layers[idx].input_layernorm(hidden_state).cpu()
                    attn_inputs_by_layer[idx].append(hidden_state)

                if args.enable_fsdp:
                    dist.barrier()

                if (step + 1) % CUTOFF_BATCH == 0:
                    # Save layer-wise outputs to disk
                    for layer_idx, attn_inputs in enumerate(attn_inputs_by_layer):
                        attn_inputs = torch.cat(attn_inputs, dim=0)  # attn_inputs.shape is (batch, seq_len, hidden_size)
                        fpath = join(data_dir, f'attn_inputs-l={layer_idx:0{max_layer_digits}d}-s={split}-b={step:0{max_digits}d}.pt')
                        torch.save(attn_inputs, fpath)
                    if rank == 0 or not args.enable_fsdp:
                        print(f'-> Saved layer-wise tensors for {split} to {data_dir}!')
                        print(f'-> Example: {fpath}')
                    del attn_inputs_by_layer
                    attn_inputs_by_layer = [[] for _ in range(len(model.model.layers))]
        
        # Save layer-wise outputs to disk
        for layer_idx, attn_inputs in enumerate(attn_inputs_by_layer):
            attn_inputs = torch.cat(attn_inputs, dim=0)  # attn_inputs.shape is (batch, seq_len, hidden_size)
            fpath = join(data_dir, f'attn_inputs-l={layer_idx:0{max_layer_digits}d}-s={split}-b={step:0{max_digits}d}.pt')
            torch.save(attn_inputs, fpath)
        if rank == 0 or not args.enable_fsdp:
            print(f'-> Saved layer-wise tensors for {split} to {data_dir}!')
            print(f'-> Example: {fpath}')

if __name__ == "__main__":
    main()
