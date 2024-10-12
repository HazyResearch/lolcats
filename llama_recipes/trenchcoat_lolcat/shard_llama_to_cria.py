""" 
This file just needs to save out the shards for 405B. 

Notes:
- Make sure that register_buffer inv_freq persistent=True for your modelling_llama.py 
"""

from typing import Optional, Tuple
import sys
import os
from os.path import join
import copy

import argparse
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import PretrainedConfig, LlamaConfig
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaAttention, LlamaMLP, LlamaRMSNorm, LlamaConfig, 
    LlamaModel, LlamaForCausalLM, LlamaRotaryEmbedding
)

from src.utils.setup import (
    init_wandb, seed_everything, flatten_config, get_run_name_from_args,
    update_config_from_args, update_model_config_from_args,
)
from src.utils.logging import print_config, print_header
from src.model.pretrained import get_pretrained_loader

# distributed
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, StateDictType  # ours
)
from llama_recipes.configs import fsdp_config as FSDP_CONFIG

from accelerate.utils import is_xpu_available
from llama_recipes.distill_llama import setup_fsdp_config
from llama_recipes.trainer_attention import (
    train,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
)
from llama_recipes.utils.fsdp_utils import (
    hsdp_device_mesh as get_hsdp_device_mesh
)
from src.model.convert_model import toggle_attention, remove_base_attention, traverse_layers

from trenchcoat_llama import LlamaMiniModelForCausalLM, LlamaMiniDecoderLayer, LlamaMiniModel

def main():
    # ------
    # SET UP
    # ------
    args = get_args()
    if args.enable_fsdp:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
    kwargs = vars(args)
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


    # Where to save the output model checkpoints?
    checkpoint_dir = args.shard_dir
    model_name = model_config.model.pretrained_model_name_or_path
    checkpoint_dir = join(checkpoint_dir, args.model_config)
    if not os.path.isdir(checkpoint_dir) and ((args.enable_fsdp and rank == 0 and local_rank == 0) or not args.enable_fsdp):
        os.makedirs(checkpoint_dir)
        # Save individual .pt model weights in a subdirectory
        checkpoint_dir = join(checkpoint_dir, 'sharded_layers')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)

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
    name_suffix = f'in={start:0{max_digits}d}-out={end:0{max_digits}d}'
    args.run_name += f'-{name_suffix}'  # will save layer-wise checkpoints
    args.run_name = args.run_name.replace('True', '1').replace('False', '0')  # concise hacks
    if rank == 0 or not args.enable_fsdp:
        print(f"Running distill for {num_hidden_layers}; Layers {start} through {end}!")
        print(f"{args.run_name=}")

    dtype = getattr(torch, model_config['model']['torch_dtype'])
    if rank == 0 or not args.enable_fsdp:
        print_header('Pretrained Model Config')
        print(pretrained_model_config)

    try:  
        # Test HF transformers version
        teacher_attn = LlamaAttention(pretrained_model_config, layer_idx=args.layer_idx)
    except KeyError:  # Might error on RoPE type due to HF transformer version
        pretrained_model_config = pretrained_model_config.to_dict()
        pretrained_model_config['rope_scaling']['type'] = pretrained_model_config['rope_scaling']['rope_type']
        pretrained_model_config = LlamaConfig.from_dict(pretrained_model_config)
        # teacher_attn = LlamaAttention(pretrained_model_config, layer_idx=args.layer_idx)

    mini_config = copy.deepcopy(pretrained_model_config).to_dict()
    mini_config['num_hidden_layers'] = args.layers_per_model
    mini_config['attn_implementation'] = 'eager'
    mini_config['low_cpu_mem_usage'] = True
    mini_config = LlamaConfig.from_dict(mini_config)
    
    # Load relevant model weights from memory for the mini student and teacher models
    if rank == 0 or not args.enable_fsdp:
        print(f"Now saving the shards!")

    # load the model
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

    if rank == 0: print(f"Loaded the model.")
        
    # Save pretrained Transformer weights
    if rank == 0:
        print(model_config) 
        print(mini_config)
    
    # SA: important for fitting the model at large block/cria sizes
    with torch.device('meta'):
        mini_llama = LlamaMiniModelForCausalLM(mini_config).to(torch.bfloat16)
    mini_llama = mini_llama.to_empty(device='cpu')
    if rank == 0:
        print(mini_llama)
        print(f"Initialized mini llama.")
    
    mini_init = {}
    for i in range(args.layers_per_model): mini_init[i] = False
    with torch.no_grad():
        first = 0
        for layer_idx, layer in enumerate(tqdm(traverse_layers(model))):
            print(f'Saving layer attentions to {checkpoint_dir}...')
            pretrained_fname = model_config['model']['pretrained_model_name_or_path'].replace('/', '_')
            print(pretrained_fname)
            if layer_idx == 0 and rank == 0: 
                print(layer.state_dict().keys())
            
            mini_init[layer_idx % args.layers_per_model] = True
            mini_llama.model.layers[layer_idx % args.layers_per_model].load_state_dict(layer.state_dict())

            if (layer_idx + 1) % args.layers_per_model == 0: 
                if rank == 0: 
                    print(f"{layer_idx=}")

                pretrained_fname = (
                    join(checkpoint_dir, pretrained_fname) + 
                    f'-in={first:0{max_digits}d}-out={layer_idx:0{max_digits}d}.pt'
                )  

                if rank == 0 or not args.enable_fsdp:
                    print(f"Initialized?\n{mini_init=}")
                    torch.save(mini_llama.state_dict(), pretrained_fname)
                    print(f"Saved to: {pretrained_fname}!")

                first = layer_idx + 1
                del mini_llama
                if rank == 0: print(f"Deleting and making a new one.")
                with torch.device('meta'):
                    mini_llama = LlamaMiniModelForCausalLM(mini_config).to(torch.bfloat16)
                mini_llama = mini_llama.to_empty(device='cpu')
                for i in range(args.layers_per_model): mini_init[i] = False

    del model
    print(f"Deleted model.")

    # Load relevant model weights for the teacher
    print(f"Checking that shards saved correctly...")
    teacher_mini_llama = LlamaMiniModelForCausalLM(mini_config)
    start, end = args.layer_idx, args.layer_idx + args.layers_per_model
    with torch.no_grad():
        pretrained_fname = model_config['model']['pretrained_model_name_or_path'].replace('/', '_')
        pretrained_fname = join(checkpoint_dir, 'sharded_layers', pretrained_fname) + f'-{name_suffix}.pt' 
        teacher_mini_llama.load_state_dict(torch.load(pretrained_fname))
        if rank == 0 or not args.enable_fsdp:
            print_header('All teacher weights loaded successfully')

if __name__ == '__main__':
    main()
    print("Thanks for washing my dishes")
    