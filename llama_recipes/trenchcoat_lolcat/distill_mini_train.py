"""
For block-by-block attention transfer.

Alternate way to do things where we convert a block of Llama decoder layers into linear attention equivalents

This lets us linearize big models in a decentralized manner without interconnect. 
Just take a block of layers and train. For instance, the following four commands could be used to perform attention transfer on the 32 attention layers of Llama 3.1 8B, in blocks of 8 layers each:

python distill_llama_mini.py \
--model_config distill_llama3_8b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean_layer_xent1_mse1000 \
--lk_zero_init --lr 1e-3 \
--layer_idx 0 --layers_per_model 8 --device 0 \
--verbose --seed 0 --replicate 0 

python distill_llama_mini.py \
--model_config distill_llama3_8b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean_layer_xent1_mse1000 \
--lk_zero_init --lr 1e-3 \
--layer_idx 8 --layers_per_model 8 --device 0 \
--verbose --seed 0 --replicate 0

python distill_llama_mini.py \
--model_config distill_llama3_8b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean_layer_xent1_mse1000 \
--lk_zero_init --lr 1e-3 \
--layer_idx 16 --layers_per_model 8 --device 0 \
--verbose --seed 0 --replicate 0

python distill_llama_mini.py \
--model_config distill_llama3_8b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean_layer_xent1_mse1000 \
--lk_zero_init --lr 1e-3 \
--layer_idx 24 --layers_per_model 8 --device 0 \
--verbose --seed 0 --replicate 0
"""


from typing import Optional, Tuple, Union, List
import sys
import os
from os.path import join
import copy

import argparse
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import PretrainedConfig
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaAttention, LlamaConfig
)
from transformers.modeling_outputs import (
    CausalLMOutputWithPast
)

from src.utils.setup import (
    init_wandb, seed_everything, flatten_config, get_run_name_from_args,
    update_config_from_args, update_model_config_from_args,
)
from src.utils.logging import print_config, print_header
from src.dataloaders import load_data
from src.trainer import get_trainer, get_optimizer, get_scheduler
from src.finetune import prepare_finetune_configs, get_finetuner

from src.model.pretrained import get_pretrained_loader
from src.model.load_model import load_and_convert_attns, load_and_convert_finetune
from src.model.convert_model import toggle_attention, remove_base_attention, traverse_layers
from src.model.utils import count_parameters
from src.model.convert_model import get_attention

from trenchcoat_args import get_args
from trenchcoat_llama import LlamaMiniDecoderLayer, LlamaMiniModelForCausalLM, LlamaMiniModel

def main():
    # ------
    # SET UP
    # ------
    args = get_args()

    CHECKPOINT_DIR = args.shard_dir
    DATA_DIR = args.attentions_data_dir

    # Where to save the output model checkpoints?
    checkpoint_dir = CHECKPOINT_DIR
    checkpoint_dir = join(checkpoint_dir, args.model_config)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # Save individual .pt model weights in a subdirectory
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # SA: What does this do?
    args.results_dir = join(args.results_dir, args.model_config)
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)
    seed_everything(args.seed)

    # Load distillation + (hedgehog) attention configs
    distill_config_path = join('./configs/experiment', f'{args.distill_config}.yaml')
    distill_config = OmegaConf.load(distill_config_path)
    distill_config = update_config_from_args(distill_config, args)

    model_config_path = join('./configs/model', f'{args.model_config}.yaml')
    model_config = OmegaConf.load(model_config_path)
    model_config = update_model_config_from_args(model_config, args)


    # Get data directory for layer-wise input tensors
    dataset_name = distill_config.dataset.name
    if dataset_name == "redpajama_sample_contig": 
        from get_data import load_data_redpajama_contig as load_data
    elif dataset_name == "redpajama_sample":
        from get_data import load_data_redpajama as load_data
    elif "alpaca" in dataset_name:
        from get_data import load_data_alpaca as load_data
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    cache_dir = DATA_DIR
    model_name = model_config.model.pretrained_model_name_or_path.replace('/', '_')

    # training on one gpu
    rank = 0

    if rank == 0 or not args.enable_fsdp:
        data_dir = cache_dir    # SA Flag
        print(f"Looking for data in {data_dir}")
        assert os.path.exists(data_dir), print(f"{data_dir} does not exist!")
        print(f"Nice, {data_dir} exists!")

    # Update data tokenizer to match model
    if getattr(distill_config.dataset, 'pretrained_model_config', None) is not None:
        for k in ['pretrained_model_name_or_path', 'cache_dir']:
            distill_config.dataset.pretrained_model_config[k] = model_config.model[k]

    # Update optimizer if specified
    if 'optimizer' in model_config:
        for k, v in model_config.optimizer.items():
            distill_config.optimizer[k] = v
    
    print_header('Distillation Config')
    print_config(distill_config)
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
    print(f"Running distill for {num_hidden_layers}; Layers {start} through {end}!")
    print(f"{args.run_name=}")

    # WandB logging
    wandb = init_wandb(args)
    if wandb is not None:
        distill_config['model'] = model_config  # Combine for logging
        _flattened = {'model': model_config,
                      'model_config': args.model_config,  # config file names
                      'distill_config': args.distill_config,
                      'finetune_config': args.finetune_config,
                      'distill_checkpoint': args.load_distill_checkpoint,
                      'finetune_checkpoint': args.load_finetune_checkpoint,
                      'replicate': args.replicate}
        flatten_config(OmegaConf.to_container(distill_config), _flattened, '')
        wandb.config.update(_flattened)

    dtype = getattr(torch, model_config['model']['torch_dtype'])
    print_header('Pretrained Model Config')
    print(pretrained_model_config)

    try:  # Test HF transformers version
        teacher_attn = LlamaAttention(pretrained_model_config, layer_idx=args.layer_idx)
    except KeyError:  # Might error on RoPE type due to HF transformer version
        pretrained_model_config = pretrained_model_config.to_dict()
        pretrained_model_config['rope_scaling']['type'] = pretrained_model_config['rope_scaling']['rope_type']
        pretrained_model_config = LlamaConfig.from_dict(pretrained_model_config)
        # teacher_attn = LlamaAttention(pretrained_model_config, layer_idx=args.layer_idx)

    mini_config = copy.deepcopy(pretrained_model_config).to_dict()
    mini_config['num_hidden_layers'] = args.layers_per_model
    mini_config['attn_implementation'] = 'eager'
    mini_config = LlamaConfig.from_dict(mini_config)
    model_config.model.attn_implementation = 'eager'
    print(mini_config)
    
    print("Starting to load the weights for the chunk.")
    # Load relevant model weights from memory for the mini student and teacher models
    with torch.device('meta'):
        # If they already exist in ./checkpoints... then just load that
        teacher_mini_llama = LlamaMiniModelForCausalLM(mini_config).to(torch.bfloat16)
    teacher_mini_llama = teacher_mini_llama.to_empty(device='cpu')
    print("Got teacher template, now loading the weights.")

    # Come back to this.
    with torch.no_grad():
        pretrained_fname = model_config['model']['pretrained_model_name_or_path'].replace('/', '_')
        pretrained_fname = join(checkpoint_dir, pretrained_fname) + f'-{name_suffix}.pt'

        ckpt = torch.load(pretrained_fname, map_location='cpu')
        print(ckpt.keys())
        teacher_mini_llama.load_state_dict(ckpt, assign=True)
        print_header('All teacher weights loaded successfully')
    for p in teacher_mini_llama.parameters():   # Freeze all layers
        p.requires_grad = False

    student_mini_llama = copy.deepcopy(teacher_mini_llama)
    student_mini_llama = load_and_convert_attns(student_mini_llama, model_config,
                                                attention_type=None, # specified in model_config,
                                                checkpoint_path=None,
                                                print_model=args.verbose,
                                                train_attention=True)[0]

    
    print(f'-> Loaded pretrained attention from {pretrained_fname}!')

    device = torch.device(f'cuda:{args.device}')
    student_mini_llama = student_mini_llama.to(device, dtype=dtype)
    student_mini_llama.to(device) 

    if args.verbose:
        print_header(f'*** Initial Layer {args.layer_idx} ***')
        print(student_mini_llama)
        print_header('*** Trainable Parameters ***')
        count = 0
        for n, p in student_mini_llama.named_parameters():
            if p.requires_grad:
                print(f'├── {n} (requires_grad = {p.requires_grad}, dtype = {p.dtype})')
                count += 1
        if count == 0:
            print('(none)')

    # ---------------------------
    # Stage 1: Attention Transfer
    # ---------------------------
    if args.load_distill_checkpoint is None:
        print(f"Loading the data frm {data_dir}")
        dataloaders = load_data(data_dir, args.layer_idx, max_layer=num_hidden_layers, 
                                **distill_config.dataloader)
        train_loader = dataloaders['train']
        eval_loader  = dataloaders['validation']
        print(f"Done loading the data")

        # Log some stats
        distill_config.model_train_params = count_parameters(student_mini_llama, requires_grad=True)
        distill_config.model_total_params = count_parameters(student_mini_llama, requires_grad=False)
        pct_trainable = distill_config.model_train_params / distill_config.model_total_params

        print_header('*** Distillation Parameter Counts ***')
        print(f'├── Number training to distill:  {distill_config.model_train_params}')
        print(f'├── Number of total parameters:  {distill_config.model_total_params}')
        print(f'├── Percent training to distill: {pct_trainable * 100:.3f}%')

        # Get optimizer and scheduler
        optimizer = get_optimizer(model=student_mini_llama, **distill_config.optimizer)
        scheduler = get_scheduler(optimizer=optimizer, **distill_config.lr_scheduler)    
            
        # Load trainer 
        for arg, argv in distill_config.trainer.items():
            if arg != 'name':
                setattr(args, arg, argv)
        for _config in ['dataloader', 'optimizer', 'lr_scheduler']:
            setattr(args, _config, OmegaConf.to_container(getattr(distill_config, _config)))
            
        print(f"Getting trainer: {distill_config.trainer.name}")
        OurTrainer = get_trainer(distill_config.trainer.name)
        trainer = OurTrainer(model=student_mini_llama, 
                             layer_idx=args.layer_idx,
                             args=args,
                             train_loader=train_loader,
                             eval_loader=eval_loader,
                             optimizer_and_scheduler=(optimizer, scheduler),
                             device=args.device,
                             wandb=wandb,
                             checkpoint_suffix='_distill',
                             save_results=False,
                             **distill_config.trainer)

        # Train / distill model
        print_header('*** Distilling Attentions ***')
        print(f'├── Experiment name: {args.run_name}')
        print(f'├── Device: {args.device}')
        print(f'├── Seed: {args.seed}')
        student_mini_llama = toggle_attention(student_mini_llama, train=True)

        student_mini_llama = trainer.train()
        args.load_distill_checkpoint = trainer.best_val_checkpoint_path  # saved here
    else:
        print(f"Loding stage 1 model from path: {args.load_distill_checkpoint}\n\n")
        with torch.no_grad():
            student_mini_llama.load_state_dict(
                torch.load(args.load_distill_checkpoint)['model_state_dict'], strict=False,)

    # # No xent calcs in the second stage for mem savings.
    # def toggle_mem_save(model, mem_save: bool = False):
    #     for layer in traverse_layers(model):
    #         layer.self_attn.mem_save = mem_save
    #     return model
    # toggle_mem_save(student_mini_llama, mem_save=True)

    # Reset the wandb
    wandb.finish() # End the first run
    print_header('*** Done training ***')
    print('--> Saved Checkpoints:')
    print(f'--attn_mlp_checkpoint_path {args.load_distill_checkpoint} \\')

if __name__ == '__main__':
    main()
    print("Thanks for washing my dishes")

    