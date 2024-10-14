""" 
This script takes the distributed FSDP files from the fine-tuning stage and assembles a single .pt file, so that we can plug that into our evaluation code.
"""

import sys
sys.path.append("../../lolcats")

import os
from os.path import join

import argparse
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import ( LlamaConfig )
import torch.optim as optim
from omegaconf import OmegaConf
from transformers.models.llama.modeling_llama import LlamaDecoderLayer as DecoderLayer

# Distributed arguments
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType
)

import torch.distributed as dist

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.policies import apply_fsdp_checkpointing
from llama_recipes.utils.fsdp_utils import fsdp_auto_wrap_policy
from accelerate.utils import is_xpu_available
from llama_recipes.utils.fsdp_utils import (
    hsdp_device_mesh as get_hsdp_device_mesh
)

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
    load_sharded_model_single_gpu,
)

from src.utils.setup import (
    init_wandb, seed_everything, flatten_config, get_run_name_from_args,
    update_config_from_args, update_model_config_from_args,
)
from src.utils.logging import print_config, print_header
from src.model.pretrained import get_pretrained_loader

from src.model.convert_model import toggle_attention, remove_base_attention, traverse_layers
from src.model.load_model import load_and_convert_attns, load_and_convert_finetune

from src.trainer import get_scheduler
from src.finetune import prepare_finetune_configs  # get_finetuner
from llama_recipes.distill_llama import (
    setup_wandb, get_args,  get_run_name_from_checkpoint,
    get_dataloaders, setup_fsdp_config
)

from trenchcoat_args import get_args

def setup_fsdp_config(config, args, checkpoint_name: str = 'finetune', output_dir: str = None):
    """
    Hacky arguments for llama-recipes training function
    -> hardcoded save path
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
    config.output_dir = args.checkpoint_dir if output_dir is None else output_dir
    config.save_metrics = not args.no_wandb
    config.num_epochs = getattr(config.trainer, 'num_train_epochs', None)
    config.num_train_steps = getattr(args, 'num_train_steps', None)  # exit training loop early for debugging
    config.eval_steps = getattr(config.trainer, 'eval_steps', None),  # config.trainer.eval_steps  # how many gradient updates before evaluating
    return config


def main():
    # ------
    # SET UP
    # ------
    args = get_args()

    CHECKPOINT_DIR_405B =  args.checkpoint_dir
    CHECKPOINT_MODEL_CONFIG = args.checkpoint_model_config
    # CHECKPOINT_MODEL_CONFIG = 'llama3_1_405b/distill_llama3_1_405b_lk_smd_wtk64_fd64_w01'
    CHECKPOINT_MODEL_CONFIG = 'llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01'

    if args.enable_fsdp:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])

    # Where to save the output model checkpoints?
    args.checkpoint_dir = join(args.checkpoint_dir, args.model_config)
    if not os.path.isdir(args.checkpoint_dir) and ((args.enable_fsdp and rank == 0 and local_rank == 0) or not args.enable_fsdp):
        os.makedirs(args.checkpoint_dir)

    # Save individual .pt model weights in a subdirectory
    args.checkpoint_dir = join(args.checkpoint_dir, 'sharded_layers')
    if not os.path.isdir(args.checkpoint_dir) and ((args.enable_fsdp and rank == 0 and local_rank == 0) or not args.enable_fsdp):
        os.makedirs(args.checkpoint_dir)

    args.results_dir = join(args.results_dir, args.model_config)
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)
    seed_everything(args.seed)

    # Load distillation + (hedgehog) attention configs
    distill_config_path = join('./configs/experiment', f'{args.distill_config}.yaml')
    distill_config = OmegaConf.load(distill_config_path)
    distill_config = update_config_from_args(distill_config, args)
    distill_config = setup_fsdp_config(distill_config, args, 'distill')  # patch 

    fsdp_config = FSDP_CONFIG()
    if is_xpu_available():
        torch.xpu.manual_seed(distill_config.seed)
    torch.manual_seed(distill_config.seed)
    import random
    random.seed(distill_config.seed)

    from llama_recipes.utils.config_utils import (update_config,get_dataloader_kwargs,)
    update_config((fsdp_config), **vars(args))

    if args.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
    
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
            _run_name = args.run_name
            kwargs = vars(args)
            kwargs['run_name'] = _run_name
            if args.final_finetune_config is not None:  # Update checkpoint for e2e finetune and lora loading
                kwargs['run_name'] += f'-ef={args.final_finetune_config}'
            kwargs['run_name'] += f'-ft_lora={args.load_finetuned_loras}'.replace('True', '1').replace('False', '0')
            wandb_run = setup_wandb(distill_config, fsdp_config, **kwargs,
                                    project=args.project_name, entity=args.wandb_entity)

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
    model_config.model.low_cpu_mem_usage = True

    if rank == 0 or not args.enable_fsdp:
        print_header('Model Config')
        print_config(model_config)

    # Get model class and configs for layer instantiating
    pretrained_model_config = LlamaConfig.from_pretrained(model_config['model']['pretrained_model_name_or_path'])
    pretrained_model_class = pretrained_model_config.architectures[0]
    transformers_module = __import__('transformers')
    pretrained_model_class = getattr(transformers_module, pretrained_model_class)  # e.g, LlamaForCausalLM

    # Final run name / checkpoint naming setup
    num_hidden_layers = pretrained_model_config.num_hidden_layers  # e.g., 32 for Llama 8B
    max_digits = len(str(num_hidden_layers))
    start, end = args.layer_idx, args.layer_idx + args.layers_per_model - 1
    args.run_name = args.run_name.replace('True', '1').replace('False', '0')  # concise hacks
    if rank == 0 or not args.enable_fsdp:
        print(f"Running distill for {num_hidden_layers}; Layers {start} through {end}!")
        print(f"{args.run_name=}")

    dtype = getattr(torch, model_config['model']['torch_dtype'])
    if rank == 0 or not args.enable_fsdp:
        print_header('Pretrained Model Config')
        print(pretrained_model_config)

    # Step 1. Load the pretrained model and tokenizer.
    if rank == 0 or not args.enable_fsdp:
        print(model_config)
    model_loader = get_pretrained_loader(**model_config.model,
                                         huggingface_token=args.huggingface_token)
    model = model_loader.load(model_type='softmax')
    if rank == 0 or not args.enable_fsdp:
        print(model)
    if args.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)
    for p in model.parameters():   # Freeze all layers
        p.requires_grad = False
    model.eval()

    tokenizer = model_loader.load_tokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    if rank == 0: print(f"Loaded the model.")
        
    # Step 2. Linearized template.
    model = load_and_convert_attns(
        model,
        model_config,
        attention_type=None, 
        checkpoint_path=None,
        print_model=args.verbose,
        train_attention=False)[0]
    
    # Step 3. Loop through the saved checkpoints.
    def check_state_dict_keys(_keys, layer_idx):
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

    def rename_state_dict(rename_dict, start_layer_idx): 
        new_state_dict = {}
        for k, v in rename_dict.items(): 
            if "layers" in k:
                k_name = k.split("layers.")[-1].split(".")[0]
                k_idx = int(k_name)
                new_k_idx = k_idx + start_layer_idx
                new_k_name = k.replace(k_name, str(new_k_idx))
                new_state_dict[new_k_name] = v
                if start_layer_idx > 9 and start_layer_idx < 18: 
                    print(f"Renaming {k} to {new_k_name}")
            else:
                new_state_dict[k] = v
        return new_state_dict

    if not args.no_distill:
        with torch.no_grad():
            first = 0
            for layer_idx, layer in enumerate(tqdm(traverse_layers(model))):
                load_file_name = join(CHECKPOINT_DIR_405B, f'dl-d={args.distill_config}-m={CHECKPOINT_MODEL_CONFIG}-f={args.finetune_config}')
                load_file_name += f'-s={args.seed}-se={args.seed}-re={args.replicate}'
                max_digits = len(str(num_hidden_layers))
                start, end = first, first + (args.layers_per_model - 1)
                name_suffix = f'in={start:0{max_digits}d}-out={end:0{max_digits}d}'
                load_file_name += f'-{name_suffix}_distill.pt' 
                load_file_name = load_file_name.replace('True', '1').replace('False', '0')  # concise hacks
                
                if (layer_idx + 1) % args.layers_per_model == 0:
                    if rank == 0 or not args.enable_fsdp:
                        print(f'Loading layer attentions from {CHECKPOINT_DIR_405B}...')
                        mini_weights = torch.load(load_file_name)['model_state_dict']
                        mini_weights = rename_state_dict(mini_weights, first)
                        _keys = model.load_state_dict(mini_weights, strict=False)
                        check_state_dict_keys(_keys, first)
                    first = layer_idx + 1
    
    # Step 4. Add finetuning parameters. 
    final_finetune_config, args = prepare_finetune_configs(args, model_config, 
                                                           args.final_finetune_config)
    final_finetune_config = setup_fsdp_config(final_finetune_config, args, 'finetune', output_dir='./') 

    args.finetune_lr = None 
    model, _ = load_and_convert_finetune(model, final_finetune_config,
                                         checkpoint_path=None,
                                         print_model=args.verbose,
                                         merge_loras=False,
                                         peft_gradient_checkpointing=not args.no_peft_grad_ckpt,
                                         rank=rank)
    
    # Actual final run name / checkpoint naming
    if (args.final_finetune_config is not None and 
        f'-ef={args.final_finetune_config}' not in args.run_name):  # Update checkpoint for e2e finetune and lora loading
        args.run_name += f'-ef={args.final_finetune_config}'
        args.run_name += f'-ft_lora={args.load_finetuned_loras}'.replace('True', '1').replace('False', '0')
    if args.no_distill:
        args.run_name += '-no_distill'

    if args.load_finetune_checkpoint is not None:
        model = load_sharded_model_single_gpu(model, model_path=args.load_finetune_checkpoint, cfg=final_finetune_config, rank=rank)
    else:
        model = load_sharded_model_single_gpu(model, model_path=None, cfg=final_finetune_config, rank=rank)
    
    
    if rank == 0:  # debugging
        print_header('** Sanity check model weights **')
        for n, p in model.named_parameters():
            if ('layers.0.' in n and 'base_attn' not in n and 
                '.0.mlp.' not in n and '.block_sparse_moe' not in n):
                print(f'-> {n}:\n', p)

    args.run_name = args.run_name.replace('llama3_1_405b/', '')
    args.run_name = args.run_name.replace('llama3_1_70b/', '')
    args.run_name = args.run_name.replace('llama3_1_8b/', '')

    print(model.config)
    if rank == 0:
        with torch.no_grad():
            state_dict = model.state_dict()
            keys_to_keep = [key for key in state_dict.keys() if 'lora' in key or 'window_factors' in key or 'feature_map' in key]
            new_state_dict = {key: state_dict[key] for key in keys_to_keep}
            torch.save(new_state_dict, f'ckpt_lora-{args.run_name}.pt')

        print_header('*** Weights in state_dict ***')
        for k in torch.load(f'ckpt_lora-{args.run_name}.pt'):
            if int(k.split('layers.')[-1].split('.')[0]) < 1:
                print(k)
        print('-> Checkpoints saved to:', f'ckpt_lora-{args.run_name}.pt')    
        

if __name__ == '__main__':
    main()
    print("Thanks for washing my dishes")

