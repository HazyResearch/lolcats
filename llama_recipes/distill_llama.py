# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
"""
Train learnable linear attentions. Rough adaptation of llama_recipes script for distillation
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
    # ShardingStrategy,
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
# from llama_recipes.utils.dataset_utils import get_preprocessed_dataset

from llama_recipes.utils.fsdp_utils import (
    hsdp_device_mesh as get_hsdp_device_mesh
)
from llama_recipes.trainer_attention import (
    train as _train_normal,
    # freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)
# Ours
from llama_recipes.model_checkpointing.distill_checkpoint_handler import (
    load_model_sharded,
    # load_model_checkpoint,
    # save_model_checkpoint,
)
# from llama_recipes.trainer_attention_chunked import train as train_chunked
# from torch.distributed.optim import DistributedOptimizer

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
    # load_and_convert_finetune
)
from src.model.convert_model import toggle_attention


def get_run_name_from_checkpoint(checkpoint_path: str) -> str:
    """Return a string describing the run from checkpoint path"""
    name = []
    for s in checkpoint_path.split('/')[-1].split('-'):
        try:
            s = s.split('=')
            s = ''.join([c[0] for c in s[1].split('_')])
            name.append(s)
        except IndexError:
            pass
    return ''.join(name)


def get_args():
    """Get attention transfer args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default='lolcats')
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--distill_config", type=str, default=None)
    parser.add_argument("--finetune_config", type=str, default=None)
    parser.add_argument("--eval_config", type=str, default=None)

    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--load_distill_checkpoint", type=str, default=None)
    parser.add_argument("--load_finetune_checkpoint", type=str, default=None)
    parser.add_argument("--resume_distill", action='store_true', default=None)
    parser.add_argument("--resume_finetune", action='store_true', default=None)

    # Override default configs
    # Feature map / model
    parser.add_argument("--attention_type", type=str, default=None)
    parser.add_argument("--learned_kernel", type=str, default=None)
    parser.add_argument("--lk_skip_connection", action='store_true', default=None)
    parser.add_argument("--lk_zero_init", action='store_true', default=None)
    parser.add_argument("--tie_qk_kernels", action='store_true', default=None)
    parser.add_argument("--train_qk", action='store_true', default=None)
    parser.add_argument("--state_chunk_len", type=int, default=None)
    
    # Training
    ## Distributed training / Llama recipes
    parser.add_argument("--enable_fsdp", action='store_true', default=None)
    parser.add_argument("--low_cpu_fsdp", action='store_true', default=None)
    parser.add_argument("--pure_bf16", action='store_true', default=None)
    parser.add_argument("--fsdp_activation_checkpointing", action='store_true', default=None)
    
    ## Hyperparameters
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--optim", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--no_peft_grad_ckpt", action='store_true', default=None)

    # Finetuning
    parser.add_argument("--finetune_lr", type=float, default=None)
    parser.add_argument("--finetune_attn_mlps", action='store_true', default=None)
    
    # Dataloading
    parser.add_argument("--dataset_chunk_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)

    # Evaluation
    parser.add_argument("--no_init_eval", action='store_true', default=False)
    parser.add_argument("--eval_steps", type=int, default=None)

    # Miscellaneous
    parser.add_argument("--huggingface_token", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default='./checkpoints')
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action='store_true', default=None)
    parser.add_argument("--no_cuda", action='store_true', default=None)
    parser.add_argument("--no_wandb", action='store_true', default=None)
    parser.add_argument("--wandb_entity", type=str, default='hazy-research')
    parser.add_argument("--debug", action='store_true', default=None)
    parser.add_argument("--num_train_steps", type=int, default=-1)

    # DEMO
    ## Generation
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    ## Miscellaneous
    parser.add_argument("--benchmark", action='store_true', default=False)
    parser.add_argument("--print_model", action='store_true', default=False)

    args = parser.parse_args()

    distill_name = args.distill_config
    finetune_name = args.finetune_config
    # Alternative default naming
    # if args.load_distill_checkpoint is not None and args.load_distill_checkpoint != 'default':
    #     distill_name = get_run_name_from_checkpoint(args.load_distill_checkpoint)
    # else:
    #     distill_name = args.distill_config
    # if args.load_finetune_checkpoint is not None and args.load_finetune_checkpoint != 'default':
    #     finetune_name = get_run_name_from_checkpoint(args.load_finetune_checkpoint)
    # else:
    #     finetune_name = args.finetune_config
    # if args.load_finetune_long_checkpoint is not None:
    #     finetune_long_name = get_run_name_from_checkpoint(args.load_finetune_long_checkpoint)
    # else:
    #     finetune_long_name = args.finetune_long_config

    args.run_name = f'dl-d={distill_name}-m={args.model_config}-f={finetune_name}'
    if args.no_peft_grad_ckpt is not None:
        args.run_name += f'-npgc={args.no_peft_grad_ckpt}'
    if args.fsdp_activation_checkpointing is not None:
        args.run_name += f'-fac={args.fsdp_activation_checkpointing}'
    # if args.dataset_chunk_size is not None:
    #     args.run_name += f'-dcs={args.dataset_chunk_size}'
    # args.run_name += f'-s={args.seed}'
    
    if args.debug:
        args.run_name += '-debug'
    
    args.run_name = args.run_name.replace('True', '1').replace('False', '0')  # concise hacks
    return args


def setup_wandb(train_config, fsdp_config, run_name = None, **kwargs):
    """
    Setup WandB for logging
    """
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    from llama_recipes.configs import wandb_config as WANDB_CONFIG
    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(name=run_name, **init_dict)
    run.config.update(train_config)
    run.config.update(fsdp_config, allow_val_change=True)
    return run


def get_dataloaders(train_config, tokenizer, no_shuffle_train: bool = False):
    """Return tuple of train_loader, eval_loader, updated train_config"""
    dataloaders  = load_data(train_config.dataset, train_config.dataloader)
    train_loader = dataloaders[train_config.trainer.train_split]
    eval_loader  = dataloaders[train_config.trainer.val_split]

    # Load and preprocess the dataset for training and validation
    dataset_train = train_loader.dataset
    dataset_eval = eval_loader.dataset

    if getattr(dataset_train, 'metric', None) is not None:
        metric = dataset_train.metric
    else:
        metric = None

    batching_strategy = getattr(train_config, 'batching_strategy', 'packing')
    train_config.batching_strategy = batching_strategy
    train_config.batch_size_training = train_config.dataloader.batch_size
    train_config.val_batch_size = train_config.dataloader.batch_size
    
    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer,   # shuffle=mode=="train",
                                            "train_no_shuffle" if no_shuffle_train else "train")  # hacky patch
    val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_eval, tokenizer, "val")
    # val_dl_kwargs['collate_fn'] = eval_loader.collate_fn

    # Create DataLoaders for the training and validation dataset
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.dataloader.num_workers,  # train_config.num_workers_dataloader,
        pin_memory=False,  # True,
        **train_dl_kwargs,
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset_eval,
        num_workers=train_config.dataloader.num_workers,
        pin_memory=False,  # True,
        **val_dl_kwargs,
    )
    return train_loader, eval_loader, train_config


def setup_fsdp_config(config, args, checkpoint_name: str = 'finetune'):
    """
    Hacky arguments for llama-recipes training function
    """
    config.seed = args.seed
    config.enable_fsdp = args.enable_fsdp
    config.low_cpu_fsdp = args.low_cpu_fsdp
    config.dist_checkpoint_root_folder = args.checkpoint_dir
    config.dist_checkpoint_folder = checkpoint_name

    config.model_name = args.run_name
    config.use_peft = False  # We have custom logic for saving PEFT modules

    if getattr(config, 'fsdp', None) is None:
        config.save_model = True
        config.run_validation = True
        config.use_fp16 = False
        config.save_model = True
        config.save_optimizer = False
        config.gradient_clipping = False
        config.gradient_clipping_threshold = 1.0
    else:
        for attr in ['save_model', 'run_validation', 'use_fp16', 'save_optimizer',
                     'gradient_clipping', 'gradient_clipping_threshold']:
            setattr(config, attr, getattr(config.fsdp, attr))
    config.output_dir = args.checkpoint_dir
    config.save_metrics = not args.no_wandb
    config.num_epochs = config.trainer.num_train_epochs
    config.num_train_steps = getattr(args, 'num_train_steps', None)  # exit training loop early for debugging
    config.eval_steps = config.trainer.eval_steps  # how many gradient updates before evaluating
    return config


def main():
    # ---------
    # 1. SET UP
    # ---------
    args = get_args()
    args.checkpoint_dir = join(args.checkpoint_dir, args.model_config)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    kwargs = vars(args)

    # if 'distill_long' in args.distill_config:
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
            print('   - Llama-recipes says "latest pytorch nightly build is required to run with low_cpu_fsdp config"')
            print("   - But who knows maybe this will work. We're just trying stuff.")
            print("   - Also if PyTorch was installed after July 1, 2023 we should be good.")
        #     raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            # "please install latest nightly.")
        model = model_loader.load(args.attention_type)
        model.state_chunk_len = model_config['attention']['state_chunk_len']
        # if rank == 0:   # Older, but now AutoModelForCausalLM.from_pretrained() handles
        #     model = model_loader.load(args.attention_type)
        #     model.state_chunk_len = model_config['attention']['state_chunk_len']
        #     # For finetuning, if weights are saved to single .pt file we should load here
        #     # -> otherwise for sharded state_dicts we load after FSDP wrapping
        # else:
        #     pretrained_config = ModelConfig.from_pretrained(**model_loader.loading_kwargs)
        #     pretrained_config.use_cache = use_cache
        #     if getattr(pretrained_config, 'rope_scaling', None) is not None:
        #         # kinda backwards, but see https://github.com/huggingface/transformers/blob/868d36d29ec132deeaaf8571b25b6a1b911d0145/src/transformers/models/llama/modeling_llama.py#L110
        #         pretrained_config.rope_scaling['type'] = 'default'  # pretrained_config.rope_scaling['rope_type']
        #     with torch.device("meta"):
        #         model = ModelClass(pretrained_config)
    else:
        model = model_loader.load(args.attention_type)
        model.state_chunk_len = model_config['attention']['state_chunk_len']

    if args.enable_fsdp and rank == 0 or not args.enable_fsdp:
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
    # 3. CONVERT + DISTILL ATTENTIONS
    # -------------------------------
    model, distill_peft_config = load_and_convert_attns(model, model_config, 
                                                        attention_type=args.attention_type, 
                                                        checkpoint_path=None,  # args.load_distill_checkpoint, 
                                                        print_model=args.verbose,
                                                        merge_loras=False,
                                                        peft_gradient_checkpointing=not args.no_peft_grad_ckpt,
                                                        train_attention=True,
                                                        rank=rank)
    model = toggle_attention(model, train=True)
    if 'lora' in args.model_config and rank == 0:  # a bit hacky, but we should name model_config to indicate peft
        model.print_trainable_parameters()
    if wandb_run and distill_peft_config is not None:
        wandb_run.config.update(distill_peft_config)

    # if rank == 0:  # debugging
    #     print_header('** Sanity check model weights **')
    #     for n, p in model.named_parameters():
    #         if ('layers.0.' in n and 'base_attn' not in n and 
    #             '.0.mlp.' not in n and '.block_sparse_moe' not in n):
    #             print(f'-> {n}:\n', p)

    hsdp_device_mesh = None
    if fsdp_config.hsdp and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
        hsdp_device_mesh = get_hsdp_device_mesh(replica_group_size=fsdp_config.replica_group_size, sharding_group_size=fsdp_config.sharding_group_size)
        print("HSDP device mesh is ready")
        
    # Setting up FSDP if enable_fsdp is enabled
    if args.enable_fsdp:
        # if not train_config.use_peft and train_config.freeze_layers:
        #     freeze_transformer_layers(train_config.num_freeze_layers)
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

        if args.load_distill_checkpoint:
            load_model_sharded(model, rank, distill_config)

    else:  # if not model_config.model.quantization and not args.enable_fsdp:
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
            if p.grad is not None:
                print(f"├────── Param shape: {p.size()}, Grad shape: {p.grad.size()}")

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=distill_config.optimizer.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=getattr(distill_config.optimizer, 'weight_decay', 0.),
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=distill_config.optimizer.lr,
            weight_decay=getattr(distill_config.optimizer, 'weight_decay', 0.),
        )
        # optimizer = optim.SGD(
        #     model.parameters(),
        #     lr=distill_config.optimizer.lr,
        #     weight_decay=getattr(distill_config.optimizer, 'weight_decay', 0.),
        # )
    # ex.) scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    scheduler = get_scheduler(optimizer=optimizer, **distill_config.lr_scheduler)

    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f'├── {n} (dtype = {p.dtype})')
        if p.grad is not None:
            print(f"├────── Param shape: {p.size()}, Grad shape: {p.grad.size()}")

    if args.verbose:
        print('-> Optimizer:', optimizer)
        print('-> Scheduler:', scheduler)

    # Get data
    train_dataloader, eval_dataloader, distill_config = get_dataloaders(distill_config, tokenizer)
    if not args.enable_fsdp or rank == 0:
        print(f"--> Training   Set Length = {len(train_dataloader.dataset)}")
        print(f"--> Validation Set Length = {len(eval_dataloader.dataset)}")

        if args.debug:
            print('-> local_rank:', local_rank)
            x = next(iter(train_dataloader))['input_ids']
            x = x.to(local_rank)
            print("-> x = next(iter(train_dataloader))['input_ids']")
            print("-> x = x.to(local_rank)")
            print('-> x.device:', x.device)
    torch.distributed.barrier()

    if (args.enable_fsdp and rank == 0) or not args.enable_fsdp:
        print_header('*** Training ***')
        if args.verbose:
            print_config(distill_config)
    # Start the training process
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
        wandb_run,
        eval_mode = args.replicate == 42,
    )
    # if not args.enable_fsdp or rank==0:
    #     [print(f'Key: {k}, Value: {v}') for k, v in results.items()]
    #     if not args.no_wandb:
    #         for k,v in results.items():
    #             wandb_run.summary[k] = v

    if not args.enable_fsdp or rank==0:
        print(model)
                
    # Test weights
    if fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
        pass  # Model checkpoint already saved
    elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
        # Load sharded weights across GPUs into model
        load_model_sharded(model, rank, distill_config)  # Test loading the sharded weights
        # save_model_checkpoint(model, None, rank, distill_config, epoch=distill_config.num_epochs)
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


if __name__ == "__main__":
    main()
