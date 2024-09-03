""" 
This file just needs to save out the shards for 405B. 

Notes:
- Make sure that register_buffer inv_freq persistent=True for your modeling_llama.py 
"""

import os
from os.path import join

import argparse
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.optim as optim

from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaDecoderLayer as DecoderLayer
)

# Distributed arguments
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

from accelerate.utils import is_xpu_available
from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.policies import apply_fsdp_checkpointing
from llama_recipes.utils.fsdp_utils import (
    fsdp_auto_wrap_policy,
    hsdp_device_mesh as get_hsdp_device_mesh
)
from llama_recipes.utils.config_utils import update_config

# Our arguments
from llama_recipes.trainer_finetune import (
    train,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    get_policies,
)
from llama_recipes.model_checkpointing.distill_checkpoint_handler import (
    load_model_sharded,
)
from llama_recipes.distill_llama import (
    setup_wandb, get_args,  # get_run_name_from_checkpoint
    get_dataloaders, setup_fsdp_config
)

from src.utils.setup import (
    seed_everything, get_run_name_from_args,
    update_config_from_args, update_model_config_from_args,
)
from src.utils.logging import print_config, print_header
from src.model.pretrained import get_pretrained_loader

from src.model.convert_model import toggle_attention, remove_base_attention, traverse_layers
from src.model.load_model import load_and_convert_attns, load_and_convert_finetune

from src.trainer import get_scheduler
from src.finetune import prepare_finetune_configs  # get_finetuner



def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default='lolcats')
    parser.add_argument("--layers_per_model", type=int)
    parser.add_argument("--layer_idx", type=int)  # specify starting layer
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--distill_config", type=str, default=None)
    parser.add_argument("--finetune_config", type=str, default=None)
    parser.add_argument("--eval_config", type=str, default=None)

    parser.add_argument("--load_finetuned_loras", action='store_true', default=False)
    parser.add_argument("--e2e_finetune_config", type=str, default=None)

    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--load_distill_checkpoint", type=str, default=None)
    parser.add_argument("--resume_distill", action='store_true', default=None)
    
    parser.add_argument("--load_finetune_checkpoint", type=str, default=None)
    parser.add_argument("--resume_finetune", action='store_true', default=None)

    # Override default configs
    # Feature map / model
    parser.add_argument("--attention_type", type=str, default=None)
    parser.add_argument("--learned_kernel", type=str, default=None)  # always
    parser.add_argument("--lk_skip_connection", action='store_true', default=None)
    parser.add_argument("--lk_zero_init", action='store_true', default=None)
    parser.add_argument("--lk_normal_init", action='store_true', default=None)
    parser.add_argument("--tie_qk_kernels", action='store_true', default=None)
    parser.add_argument("--train_qk", action='store_true', default=None)
    parser.add_argument("--state_chunk_len", type=int, default=None)
    
    # Training
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--optim", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_finetune_steps", type=int, default=None)

    parser.add_argument("--no_peft_grad_ckpt", action='store_true', default=None)
    
    ## Distributed training / Llama recipes
    parser.add_argument("--enable_fsdp", action='store_true', default=None)
    parser.add_argument("--low_cpu_fsdp", action='store_true', default=None)
    parser.add_argument("--pure_bf16", action='store_true', default=None)
    parser.add_argument("--fsdp_activation_checkpointing", action='store_true', default=None)
    parser.add_argument("--fsdp_cpu_offload", action='store_true', default=None)
    
    # Dataloading
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)

    # Evaluation
    parser.add_argument("--no_init_eval", action='store_true', default=False)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)

    # Miscellaneous
    parser.add_argument("--huggingface_token", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default='./checkpoints')
    parser.add_argument("--results_dir", type=str, default='./results')
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action='store_true', default=None)
    parser.add_argument("--no_cuda", action='store_true', default=None)
    parser.add_argument("--no_wandb", action='store_true', default=None)
    parser.add_argument("--wandb_entity", type=str, default='hazy-research')
    parser.add_argument("--debug", action='store_true', default=None)
    parser.add_argument("--no_attention_mask", action='store_true', default=None)

    args = parser.parse_args()
    args.run_name = get_run_name_from_args(args)
    return args


def check_state_dict_keys(keys: any, layer_idx: int, rank: int = 0, 
                          state_dict: dict = None, verbose: bool = True) -> None:
    """
    Check the state dict keys for unexpected and expected keys
    - keys: the output from torch.load_state_dict()
    - layer_idx: the current layer index
    """
    try:
        assert len(keys.unexpected_keys) == 0
        if rank == 0:
            print_header(f'*** All expected keys matched successfully {layer_idx} ***')
            if verbose and state_dict is not None:
                print('Keys loaded:')
                for k in state_dict:
                    print(f'├── {k}')
    except Exception as e:
        if rank == 0:
            print(e)
            print_header('*** Error: unexpected keys in checkpoint ***')
            print(f'Unexpected keys at {layer_idx}:')
            for k in keys.unexpected_keys:
                print(k)


def rename_state_dict(rename_dict: dict, start_layer_idx: int, verbose: bool = False) -> dict:
    """Rename the state dict from the mini models to match the full model"""
    new_state_dict = {}
    for k, v in rename_dict.items():
        if "layers" in k:
            k_name = k.split("layers.")[-1].split(".")[0]
            k_idx = int(k_name)
            new_k_idx = k_idx + start_layer_idx
            new_k_name = k.replace(k_name, str(new_k_idx))
            new_state_dict[new_k_name] = v
            if verbose:  # if start_layer_idx > 9 and start_layer_idx < 18: 
                print(f"-> Renaming {k} to {new_k_name}")
        else:
            new_state_dict[k] = v
    return new_state_dict


def main():
    """Main script"""
    # ------
    # SET UP
    # ------
    args = get_args()
    # args.checkpoint_dir = "/data_ephemeral/sim/sharded_layers_405b/"
    args.checkpoint_dir = join(args.checkpoint_dir, args.model_config)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    # Save individual .pt model weights in a subdirectory
    args.checkpoint_dir = join(args.checkpoint_dir, 'sharded_layers')
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    args.results_dir = join(args.results_dir, args.model_config)
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)
    seed_everything(args.seed)

    # Load distillation + (hedgehog) attention configs
    distill_config_path = join('./configs/experiment', f'{args.distill_config}.yaml')
    distill_config = OmegaConf.load(distill_config_path)
    distill_config = update_config_from_args(distill_config, args)

    for arg, argv in distill_config.trainer.items():  # legacy, should be removed
        if arg != 'name':
            setattr(args, arg, argv)
        for _config in ['dataloader', 'optimizer', 'lr_scheduler']:
            setattr(args, _config, OmegaConf.to_container(getattr(distill_config, _config)))

    model_config_path = join('./configs/model', f'{args.model_config}.yaml')
    model_config = OmegaConf.load(model_config_path)
    model_config = update_model_config_from_args(model_config, args)

     # Update data tokenizer to match model (unused in this script)
    if getattr(distill_config.dataset, 'pretrained_model_config', None) is not None:
        for k in ['pretrained_model_name_or_path', 'cache_dir']:
            distill_config.dataset.pretrained_model_config[k] = model_config.model[k]

    if args.enable_fsdp:
        if getattr(model_config.model, 'load_in_4bit', False):
            model_config.model.device_map = 'auto'
        elif getattr(model_config.model, 'load_in_8bit', False):
            model_config.model.device_map = 'auto'
        else:
            model_config.model.device_map = None  # FSDP will complain about device placement o.w.
    model_config.model.low_cpu_mem_usage = True

    # Setup FSDP if enabled
    if args.enable_fsdp:
        distill_config = setup_fsdp_config(distill_config, args, 'distill')  # patch
        fsdp_config = FSDP_CONFIG()
        update_config((fsdp_config), **vars(args))
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        # world_size = int(os.environ["WORLD_SIZE"])
    else:
        fsdp_config = FSDP_CONFIG()  # ignored
        rank = 0

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # WandB logging
    wandb_run = None
    if not args.no_wandb:
        if not args.enable_fsdp or rank == 0:
            wandb_run = setup_wandb(distill_config, fsdp_config, **vars(args),
                                    project=args.project_name, entity=args.wandb_entity)

    # Loading model
    try:
        if not os.path.exists(model_config.model.pretrained_model_name_or_path):
            print(f"Model path {model_config.model.pretrained_model_name_or_path} does not exist. Using backup path. {model_config.model.pretrained_model_name_or_path_backup}")
            model_config.model.pretrained_model_name_or_path = model_config.model.pretrained_model_name_or_path_backup
        model_config.model.pop("pretrained_model_name_or_path_backup")
    except Exception as e:
        print(f'-> Error: {e}')
        print("Model without model.pretrained_model_name_or_path_backup path")

    if rank == 0 or not args.enable_fsdp:
        print_header('Model Config')
        print_config(model_config)

    # Get model class and configs for layer instantiating
    pretrained_model_config = LlamaConfig.from_pretrained(model_config['model']['pretrained_model_name_or_path'])
    pretrained_model_class = pretrained_model_config.architectures[0]
    transformers_module = __import__('transformers')
    pretrained_model_class = getattr(transformers_module, pretrained_model_class)  # e.g, LlamaForCausalLM

    # -------------------------------------------
    # Step 1. Load pretrained model and tokenizer
    # -------------------------------------------
    if rank == 0 or not args.enable_fsdp:
        print_header('Pretrained Model Config')
        print(pretrained_model_config)
        print_header('Our Model Config')
        print(model_config)

    model_loader = get_pretrained_loader(**model_config.model,
                                         huggingface_token=args.huggingface_token)
    # Model
    model = model_loader.load(model_type='softmax')
    if rank == 0 or not args.enable_fsdp:
        print_header('Original Model')
        print(model)
    if args.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)
    for p in model.parameters():   # Freeze all layers
        p.requires_grad = False
    model.eval()
    # Tokenizer
    tokenizer = model_loader.load_tokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    
    # ---------------------------------------------------
    # Step 2. Convert attentions to linearized attentions
    # ---------------------------------------------------
    model = load_and_convert_attns(model,
                                   model_config,
                                   attention_type=None, # specified in model_config,
                                   checkpoint_path=None,
                                   print_model=args.verbose,
                                   train_attention=False)[0]
    if rank == 0 or not args.enable_fsdp:
        print_header('Converted Model')

    # ------------------------------------------
    # Step 3. Loop through the saved checkpoints
    # ------------------------------------------
    num_hidden_layers = pretrained_model_config.num_hidden_layers  # e.g., 32 for Llama 8B
    max_digits = len(str(num_hidden_layers))  # name_suffix = f'in={start:0{max_digits}d}-out={end:0{max_digits}d}'
    with torch.no_grad():
        first = 0
        for layer_idx in range(tqdm(traverse_layers(model))):
            if rank == 0 or not args.enable_fsdp:
                print(f'Loading layer attentions from {args.checkpoint_dir}...')
            load_file_name = f'{join(args.checkpoint_dir, args.run_name)}'
            start, end = first, first + (args.layers_per_model - 1)
            name_suffix = f'in={start:0{max_digits}d}-out={end:0{max_digits}d}'
            load_file_name += f'-{name_suffix}'
            load_file_name = load_file_name.replace('True', '1').replace('False', '0')  # concise hacks
            load_file_name = load_file_name + '_distill.pt'

            if (layer_idx + 1) % args.layers_per_model == 0:
                if rank == 0 or not args.enable_fsdp:
                    mini_weights = torch.load(load_file_name)['model_state_dict']
                    mini_weights = rename_state_dict(mini_weights, first)
                    _keys = model.load_state_dict(mini_weights, strict=False)
                    check_state_dict_keys(_keys, first, rank, mini_weights, verbose=args.verbose)
                first = layer_idx + 1
        args.run_name += f'-{name_suffix}'  # dealing with legacy naming

    # ---------------------------------------
    # Step 4. Add end-to-end finetuning LoRAs
    # ---------------------------------------
    e2e_finetune_config, args = prepare_finetune_configs(args, model_config,
                                                         args.e2e_finetune_config)
    e2e_finetune_config = setup_fsdp_config(e2e_finetune_config, args, 'finetune')
    model, _ = load_and_convert_finetune(model, e2e_finetune_config,
                                         checkpoint_path=None,
                                         print_model=args.verbose,
                                         merge_loras=False,
                                         peft_gradient_checkpointing=not args.no_peft_grad_ckpt,
                                         rank=rank)
    
    # ----------------------------------------------
    # Step 5. Add the LoRA weights from mini-distill
    # ----------------------------------------------
    if args.load_finetuned_loras:
        if args.enable_fsdp or rank == 0:
            print("Loading loras")
        with torch.no_grad():
            first = 0
            for layer_idx, layer in enumerate(tqdm(traverse_layers(model))):
                print(f'Loading layer loras from {args.checkpoint_dir}...')
                load_file_name = f'{join(args.checkpoint_dir, args.run_name)}'
                start, end = first, first + (args.layers_per_model - 1)
                name_suffix = f'in={start:0{max_digits}d}-out={end:0{max_digits}d}'
                load_file_name += f'-{name_suffix}' 
                load_file_name = load_file_name.replace('True', '1').replace('False', '0')  # concise hacks
                load_file_name = load_file_name + '_ft.pt'

                if (layer_idx + 1) % args.layers_per_model == 0:
                    if rank == 0 or not args.enable_fsdp:
                        mini_weights = torch.load(load_file_name)['model_state_dict']
                        mini_weights = rename_state_dict(mini_weights, first)
                        _keys = model.load_state_dict(mini_weights, strict=False)
                        check_state_dict_keys(_keys, first, rank, mini_weights, verbose=args.verbose)
                    first = layer_idx + 1

    # Ignored
    # hsdp_device_mesh = None
    # if fsdp_config.hsdp and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
    #     hsdp_device_mesh = get_hsdp_device_mesh(replica_group_size=fsdp_config.replica_group_size,
    #                                             sharding_group_size=fsdp_config.sharding_group_size)
    #     print("HSDP device mesh is ready")
    
    # Final run name / checkpoint naming setup
    if args.e2e_finetune_config is not None:  # Update checkpoint for e2e finetune and lora loading
        args.run_name += f'-ef={args.e2e_finetune_config}'
    args.run_name += f'-ft_lora={args.load_finetuned_loras}'.replace('True', '1').replace('False', '0')
    args.run_name = args.run_name.replace('True', '1').replace('False', '0')  # concise hacks
    args.run_name = args.run_name.replace(f'-{name_suffix}', '')  # remove the mini model suffix
    args.run_name = args.run_name.replace(args.model_config, ''.join([c[0] + c[-1] for c in args.model_config.split('_')]))
    args.run_name = args.run_name.replace(args.distill_config, ''.join([c[0] + c[-1] for c in args.distill_config.split('_')]))
    args.run_name = args.run_name.replace(args.finetune_config, ''.join([c[0] + c[-1] for c in args.finetune_config.split('_')]))

    # ----------------------------
    # Step 6. Wrap model with FSDP
    # ----------------------------
    if args.enable_fsdp:
        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank, model="llama")
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

        # Load distilled checkpoints
        if args.verbose and rank == 0:
            print_header('*** FSDP Model ***')
            print(model)
            print('Loading checkpoints from:', distill_config.model_name)
                    
        if rank == 0 or not args.enable_fsdp:  # debugging
            print_header('** Sanity check model weights **')
            for n, p in model.named_parameters():
                if ('layers.0.' in n and 'base_attn' not in n and 
                    '.0.mlp.' not in n and '.block_sparse_moe' not in n):
                    print(f'-> {n}:\n', p)

    # Initialize the optimizer and learning rate scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=e2e_finetune_config.optimizer.lr,
        weight_decay=getattr(distill_config.optimizer, 'weight_decay', 0.),
    )
    scheduler = get_scheduler(optimizer=optimizer, **e2e_finetune_config.lr_scheduler)

    if args.verbose and (rank == 0 or not args.enable_fsdp):
        print('-> Optimizer:', optimizer)
        print('-> Scheduler:', scheduler)
        print_header('*** FSDP MODEL ***')
        print(model)
        print_header('*** Trainable Parameters ***')
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f'├── {n} (dtype = {p.dtype})')    

    train_dataloader, eval_dataloader, e2e_finetune_config = get_dataloaders(e2e_finetune_config, tokenizer)
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

    # Step 7. Finetune the model
    if rank == 0 or not args.enable_fsdp:
        print_header('*** Training ***')
        if args.verbose:
            print_config(e2e_finetune_config)

    # Start the training process
    # max_optimizer_steps = getattr(distill_config.optimizer, 'max_optimizer_steps', None)
    results, best_checkpoint_path = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        gradient_accumulation_steps=e2e_finetune_config.trainer.gradient_accumulation_steps,
        train_config=e2e_finetune_config,  # train_config,
        fsdp_config=fsdp_config if args.enable_fsdp else None,
        local_rank=local_rank if args.enable_fsdp else None,
        rank=rank if args.enable_fsdp else None,
        wandb_run=wandb_run,
    )

    # Save best model checkpoint as single .pt file
    if fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
        pass  # Model checkpoint already saved
    elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
        # Load sharded weights across GPUs into model
        ignore_param_rule = lambda n, p: not p.requires_grad  # and 'feature_map' not in n or ('v_proj' in n or 'o_proj' in n)
        load_model_sharded(model, rank, final_finetune_config, ignore_param_rule)
        if rank == 0:  # debugging
            print_header('** Sanity check model weights **')
            for n, p in model.named_parameters():
                if ('layers.0.' in n and 'base_attn' not in n and
                    '.0.mlp.' not in n and '.block_sparse_moe' not in n):
                    print(f'-> {n}:\n', p)

    if not args.enable_fsdp or rank==0:
        for k,v in results.items():
            print(f'Key: {k}, Value: {v}')
            if not args.no_wandb:
                wandb_run.summary[f'ft_{k}'] = v
        print('-> Find weights at:', best_checkpoint_path)


if __name__ == '__main__':
    main()
