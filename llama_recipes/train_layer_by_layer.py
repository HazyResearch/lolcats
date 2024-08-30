# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
"""
Train learnable linear attentions. Rough adaptation of llama_recipes script for distillation
"""
import os
from os.path import join
import dataclasses
import random
import argparse  # ours
from tqdm import tqdm

from pkg_resources import packaging

import torch
import torch.optim as optim

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType  # ours
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
# from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    get_dataloader_kwargs,
)
from llama_recipes.utils.fsdp_utils import (
    hsdp_device_mesh as get_hsdp_device_mesh
)
from llama_recipes.trainer_layer_by_layer import (
    train,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)
from llama_recipes.model_checkpointing.distill_checkpoint_handler import ( load_model_sharded )

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
from src.dataloaders import load_data
from src.trainer import get_scheduler

from src.model.pretrained import get_pretrained_loader
from src.model.load_model import (
    load_and_convert_attns,
)
from src.model.convert_model import toggle_attention

from llama_recipes.distill_llama import get_run_name_from_checkpoint, get_args, setup_wandb, setup_fsdp_config

def get_dataloaders(
    data_path,
    layer_idx, world_size, rank, batch_size=1
):
    from torch.utils.data import Dataset, DataLoader
    class Layer2Dataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    if rank == 0: # Only the main process (rank 0) will save the data
        save_path = os.path.join(data_path, "layer2dataset.pt")
        layer2dataset = torch.load(save_path)
        layer2dataset = layer2dataset[layer_idx]

    if rank == 0:  # Only the main process (rank 0) will load the data
        save_path = os.path.join(data_path, "layer2dataset.pt")
        layer2dataset = torch.load(save_path)
        data = layer2dataset[layer_idx]
    else:
        data = None

    # Broadcast the data to all ranks
    if world_size > 1:
        if rank == 0:
            # Get the total number of tensors and their shapes
            num_tensors = len(data)
            shapes = [tensor.shape for tensor in data]
        else:
            num_tensors = None
            shapes = None

        # Broadcast num_tensors and shapes
        num_tensors = dist.broadcast_object_list([num_tensors], src=0)[0]
        shapes = dist.broadcast_object_list([shapes], src=0)[0]

        # Broadcast each tensor
        if rank != 0:
            data = [torch.zeros(shape) for shape in shapes]
        for i in range(num_tensors):
            dist.broadcast(data[i], src=0)
    else:
        # Non-distributed case: data is already loaded on the single process
        pass  # No need to do anything, data is already set


    # Create the dataset
    dataset = Layer2Dataset(data)

    # Create the dataloader
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=1,
        sampler=sampler,
        pin_memory=True
    )
    return dataloader, dataloader

def launch_training(args):
    # ---------
    # 1. SET UP
    # ---------
    layer_idx = args.layer_idx

    if args.enable_fsdp:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])

    args.checkpoint_dir = join(args.checkpoint_dir, args.model_config)
    if not os.path.isdir(args.checkpoint_dir) and ((args.enable_fsdp and rank == 0 and local_rank == 0) or not args.enable_fsdp):
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
        torch.xpu.manual_seed(distill_config.seed)
    torch.manual_seed(distill_config.seed)
    random.seed(distill_config.seed)

    if args.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if (args.enable_fsdp and rank == 0) or not args.enable_fsdp:
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
    try:
        if not os.path.exists(model_config.model.pretrained_model_name_or_path):
            print(f"Model path {model_config.model.pretrained_model_name_or_path} does not exist. Using backup path. {model_config.model.pretrained_model_name_or_path_backup}")
            model_config.model.pretrained_model_name_or_path = model_config.model.pretrained_model_name_or_path_backup
        model_config.model.pop("pretrained_model_name_or_path_backup")
    except:
        print(f"Model without model.pretrained_model_name_or_path_backup path")
        pass

    model_loader = get_pretrained_loader(**model_config.model,
                                         huggingface_token=args.huggingface_token)
    tokenizer = model_loader.load_tokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    use_cache = False if args.enable_fsdp else None

    if 'llama' in model_config.model.pretrained_model_name_or_path.lower():
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
            print('   - Llama-recipes says "latest pytorch nightly build is required to run with low_cpu_fsdp config"')
            print("   - But who knows maybe this will work. We're just trying stuff.")
            print("   - Also if PyTorch was installed after July 1, 2023 we should be good.")
        model = model_loader.load(args.attention_type)
        model.state_chunk_len = model_config['attention']['state_chunk_len']
    else:
        model = model_loader.load(args.attention_type)
        model.state_chunk_len = model_config['attention']['state_chunk_len']
    model_config.model_name = model_config.model.pretrained_model_name_or_path

    print_model_size(model, model_config, rank if args.enable_fsdp else 0)
    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if args.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    # -------------------------------
    # 3. CONVERT + DISTILL ATTENTIONS
    # -------------------------------
    from src.model.convert_model import toggle_attention, remove_base_attention, traverse_layers
    # Get teacher layer (we'll delete the entire model layer)
    teacher_layer = traverse_layers(model)[layer_idx].self_attn
    model, distill_peft_config = load_and_convert_attns(model, model_config, 
                                                        attention_type=args.attention_type, 
                                                        checkpoint_path=None,  # args.load_distill_checkpoint, 
                                                        print_model=args.verbose,
                                                        merge_loras=False,
                                                        peft_gradient_checkpointing=not args.no_peft_grad_ckpt,
                                                        train_attention=True,
                                                        rank=rank)
    # Student layer 
    model_layer = traverse_layers(model)[layer_idx].self_attn
    model_layer.train_attention = True  # Prepare for feature map training
    del model

    model = model_layer

    # Hack to avoid having to call lm_head
    if 'lora' in args.model_config and rank == 0:  # a bit hacky, but we should name model_config to indicate peft
        model.print_trainable_parameters()
    if wandb_run and distill_peft_config is not None:
        wandb_run.config.update(distill_peft_config)

    hsdp_device_mesh = None
    if fsdp_config.hsdp and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
        hsdp_device_mesh = get_hsdp_device_mesh(replica_group_size=fsdp_config.replica_group_size, sharding_group_size=fsdp_config.sharding_group_size)
        print("HSDP device mesh is ready")
        
    # Setting up FSDP if enable_fsdp is enabled
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
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=args.low_cpu_fsdp,  # train_config.low_cpu_fsdp
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if args.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)

        if args.load_distill_checkpoint:
            load_model_sharded(model, rank, distill_config)

    elif not model_config.model.quantization and not args.enable_fsdp:
        if is_xpu_available():
            model.to("xpu:0")
        elif torch.cuda.is_available():
            model.to("cuda")

    if (args.verbose and (
        (args.enable_fsdp and rank == 0) or not args.enable_fsdp)):
        print_header('*** FSDP MODEL ***')
        print(model)
        print_header('*** Trainable Parameters ***')
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f'├── {n} (dtype = {p.dtype})')

    # Initialize the optimizer and learning rate scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=distill_config.optimizer.lr,
        weight_decay=getattr(distill_config.optimizer, 'weight_decay', 0.),
    )
    scheduler = get_scheduler(optimizer=optimizer, **distill_config.lr_scheduler)

    if args.verbose:
        print('-> Optimizer:', optimizer)
        print('-> Scheduler:', scheduler)

    # Get data
    train_dataloader, eval_dataloader = get_dataloaders(
        data_path='/home/simarora/code/lolcats/data/saved_output',
        layer_idx=0, world_size=world_size, rank=rank, batch_size=1
    )
    if not args.enable_fsdp or rank == 0:
        print(f"--> Training   Set Length = {len(train_dataloader.dataset)}")
        print(f"--> Validation Set Length = {len(eval_dataloader.dataset)}")
    torch.distributed.barrier()

    if (args.enable_fsdp and rank == 0) or not args.enable_fsdp:
        print_header('*** Training ***')

    # Start the training process
    max_optimizer_steps = getattr(distill_config.optimizer, 'max_optimizer_steps', None)
    results, best_checkpoint_path = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        distill_config.trainer.gradient_accumulation_steps, # train_config.gradient_accumulation_steps,
        distill_config,  # train_config,
        fsdp_config if args.enable_fsdp else None,
        local_rank if args.enable_fsdp else None,
        rank if args.enable_fsdp else None,
        max_optimizer_steps,
        wandb_run,
        eval_mode = args.replicate == 42,
    )
    if not args.enable_fsdp or rank==0:
        print(model)
                
    # Test weights
    if fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
        pass  # Model checkpoint already saved
    elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
        # Load sharded weights across GPUs into model
        load_model_sharded(model, rank, distill_config)  # Test loading the sharded weights
        if rank == 0:  # debugging
            print_header('** Sanity check model weights **')
            for n, p in model.named_parameters():
                if ('layers.0.' in n and 'base_attn' not in n and 
                    '.0.mlp.' not in n and '.block_sparse_moe' not in n):
                    print(f'-> {n}:\n', p)
   
    if not args.enable_fsdp or rank==0:
        for k, v in results.items():
            print(f'Key: {k}, Value: {v}')
            if not args.no_wandb:
                wandb_run.summary[f'attn_{k}'] = v
        print('-> Find weights at:', best_checkpoint_path)

# def main():
#     for i in range(126): launch_training(i)

##### RAY LAUNCHER #####

import ray
import sys
from pathlib import Path

def execute_config(args):
    try: 
        project_root = Path(__file__).resolve().parent
        # Add the project root and src directory to the Python path
        sys.path.append(str(project_root))
        sys.path.append(str(project_root / 'src'))
        launch_training(args)
    except Exception as e:
        return args, e
    return args, None

def main():
    args = get_args()
    layers = 10
    num_gpus_per_job = 1
    configs = []
    for layer_idx in range(layers):
        args.layer_idx = layer_idx 
        configs.append(args)

    # ray was killing workers due to OOM, but it didn't seem to be necessary 
    os.environ["RAY_memory_monitor_refresh_ms"] = "0"
    ray.init(ignore_reinit_error=True, log_to_driver=False, runtime_env={
        "py_modules": [str(Path(__file__).resolve().parent)],
        "env_vars": {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}
    })

    # Run each script in parallel using Ray
    completed, failed = 0, 0
    print(f"Completed: {completed} ({completed / layers:0.1%}) | Total layers: {layers}")
    remote = ray.remote(num_gpus=num_gpus_per_job)(execute_config)
    futures = [remote.remote(config) for config in configs]
        
    while futures:
        complete, futures = ray.wait(futures)
        for config, error in ray.get(complete):
            if error is not None:
                failed += 1
                print(error)
            completed += 1
        print(f"Completed: {completed} ({completed / layers:0.1%} -- {failed} failed) | Total layers: {layers}")
    ray.shutdown()

if __name__ == "__main__":
    main()
