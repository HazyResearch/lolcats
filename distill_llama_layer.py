"""
Script to convert a single attention layer into a linear attention layer

python distill_llama_layer.py \
--model_config distill_llama3_1_8b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_synth_normal_llama3_1_8b_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_synth_normal_llama3_1_8b_xent1_mse1000 \
--lk_zero_init \
--verbose --seed 0 --replicate 0 \
--layer_idx 0 --device 0 --lr 1e-3

python distill_llama_layer.py \
--model_config distill_llama3_1_8b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_synth_normal_llama3_1_8b_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_synth_normal_llama3_1_8b_xent1_mse1000 \
--lk_zero_init \
--verbose --seed 0 --replicate 0 \
--layer_idx 1 --device 1 --lr 1e-3


python distill_llama_layer.py \
--model_config distill_llama3_1_8b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_synth_normal_llama3_1_8b_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_synth_normal_llama3_1_8b_xent1_mse1000 \
--lk_zero_init \
--verbose --seed 0 --replicate 0 \
--layer_idx 2 --device 2 --lr 1e-3
"""
"""
Alternate way to do things where we convert a single attention layer into a linear attention layer

This lets us linearize big models in a decentralized manner without interconnect. 
Just take a layer and train.

python distill_llama_layer.py \
--model_config distill_llama3_8b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean_layer_xent1_mse1000 \
--lk_zero_init --verbose --seed 0 --replicate 0 \
--layer_idx 0 --device 0 --lr 1e-3
"""
import sys
import os
from os.path import join

import argparse
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import PretrainedConfig, LlamaConfig

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

from transformers.models.llama.modeling_llama import LlamaAttention
from src.model.convert_model import get_attention


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default='lolcats')
    parser.add_argument("--layer_idx", type=int)  # specify the layer
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--distill_config", type=str, default=None)
    parser.add_argument("--finetune_config", type=str, default=None)
    parser.add_argument("--eval_config", type=str, default=None)

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


# ------------------------------
# Precomputed Tensor Dataloaders
# ------------------------------
class AttentionInputDataset(Dataset):
    """
    Tensor dataset for LlamaAttention model
    """
    def __init__(self, tensors: torch.Tensor):
        self.samples = tensors

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        position_ids = torch.arange(x.shape[-2])
        return {'hidden_states': x, 'position_ids': position_ids}


def load_data(data_dir: str, layer_idx: int, max_layer: int = 32, 
              **loader_kwargs: any):
    """
    Specific function to load attention input dataloaders
    """
    max_layer_digits = len(str(max_layer))

    dataloaders = {'train': None, 'validation': None}
    for split in dataloaders:
        sample_tensors = []
        for f in os.listdir(data_dir):
            # Filter and load naïvely 
            if f'-l={layer_idx:0{max_layer_digits}d}-s={split}' in f:
                sample_tensors.append(torch.load(join(data_dir, f)))
        samples = torch.cat(sample_tensors, dim=0)  # attn_inputs.shape is (batch, seq_len, hidden_size)
        _dataset = AttentionInputDataset(samples)
        _dataloader = DataLoader(_dataset, shuffle=True if split == 'train' else False,
                                 **loader_kwargs)
        dataloaders[split] = _dataloader
    return dataloaders


def main():
    # ------
    # SET UP
    # ------
    args = get_args()
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
    # args.device = torch.device('cuda')

    # Load distillation + (hedgehog) attention configs
    distill_config_path = join('./configs/experiment', f'{args.distill_config}.yaml')
    distill_config = OmegaConf.load(distill_config_path)
    distill_config = update_config_from_args(distill_config, args)

    model_config_path = join('./configs/model', f'{args.model_config}.yaml')
    model_config = OmegaConf.load(model_config_path)
    model_config = update_model_config_from_args(model_config, args)

    # Get data directory for layer-wise input tensors
    dataset_name = distill_config.dataset.name
    cache_dir = distill_config.dataset.dataset_config.cache_dir
    model_name = model_config.model.pretrained_model_name_or_path.replace('/', '_')

    rank = 0
    
    if rank == 0 or not args.enable_fsdp:
        try:
            # Example: meta-llama_Meta-Llama-3.1-70B/attn_inputs-l=31-split=train-b=0499.pt
            data_dir = join(cache_dir, dataset_name, model_name) 
        except Exception as e:
            print(f'Data directory {join(cache_dir, dataset_name, model_name)} not found.')
            print(f'Please see ./llama_recipes/save_llama_attn_inputs.py to save those tensors.')
            raise e
        
    # Update data tokenizer to match model
    if getattr(distill_config.dataset, 'pretrained_model_config', None) is not None:
        for k in ['pretrained_model_name_or_path', 'cache_dir']:
            distill_config.dataset.pretrained_model_config[k] = model_config.model[k]

    # Update optimizer if specified
    if 'optimizer' in model_config:
        for k, v in model_config.optimizer.items():
            distill_config.optimizer[k] = v

    # Update distilling trainer to reflect layer-wise
    distill_config.trainer.name = 'layer_distill_xent_mse'
    
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
    num_hidden_layers = pretrained_model_config.num_hidden_layers  # 32
    max_digits = len(str(num_hidden_layers))
    args.run_name += f'-layer={args.layer_idx:0{max_digits}d}'  # will save layer-wise checkpoints
    args.run_name = args.run_name.replace('True', '1').replace('False', '0')  # concise hacks

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

    try:  # Load individual layer from memory
        teacher_attn = LlamaAttention(pretrained_model_config, layer_idx=args.layer_idx)
        with torch.no_grad():
            pretrained_fname = model_config['model']['pretrained_model_name_or_path'].replace('/', '_')
            pretrained_fname = join(args.checkpoint_dir, pretrained_fname) + f'-attn={args.layer_idx}.pt'
            teacher_attn.load_state_dict(torch.load(pretrained_fname))
            print_header('All teacher weights loaded successfully')
        for p in teacher_attn.parameters():   # Freeze all layers
            p.requires_grad = False
        
        model_attn = get_attention(**model_config['attention'])(
            base_attn=teacher_attn, layer_idx=args.layer_idx, 
            max_layer_idx=pretrained_model_config.num_hidden_layers - 1,
            train_attention=True, remove_base_attn=True
        )  # .to(dtype=dtype)
        print(f'-> Loaded pretrained attention from {pretrained_fname}!')
    except Exception as e:  # Load entire model to disk
        print('-> Addressing exception:', e)
        # Get pretrained model
        model_config.model['device_map'] = 'cpu'
        model_loader = get_pretrained_loader(**model_config.model,
                                             huggingface_token=args.huggingface_token)
        model = model_loader.load(model_type='softmax')
        for p in model.parameters():   # Freeze all layers
            p.requires_grad = False
        model.eval()
        # Save pretrained attention weights
        with torch.no_grad():
            for layer_idx, layer in enumerate(tqdm(traverse_layers(model), desc=f'Saving layer attentions to {args.checkpoint_dir}...')):
                pretrained_fname = model_config['model']['pretrained_model_name_or_path'].replace('/', '_')
                pretrained_fname = join(args.checkpoint_dir, pretrained_fname) + f'-attn={layer_idx}.pt'
                torch.save(layer.self_attn.state_dict(), pretrained_fname)

        teacher_attn = LlamaAttention(pretrained_model_config)
        with torch.no_grad():
            pretrained_fname = model_config['model']['pretrained_model_name_or_path'].replace('/', '_')
            pretrained_fname = join(args.checkpoint_dir, pretrained_fname) + f'-attn={args.layer_idx}.pt'
            teacher_attn.load_state_dict(torch.load(pretrained_fname))
            print_header('All teacher weights loaded successfully')
        for p in teacher_attn.parameters():   # Freeze all layers
            p.requires_grad = False
        
        model_attn = get_attention(**model_config['attention'])(
            base_attn=teacher_attn, layer_idx=args.layer_idx, 
            max_layer_idx=pretrained_model_config.num_hidden_layers - 1,
            train_attention=True, remove_base_attn=True
        )
        del model

    device = torch.device(f'cuda:{args.device}')
    model_attn = model_attn.to(device, dtype=dtype)
    model_attn.device = device  # hack
    teacher_attn.eval()
    teacher_attn.to(dtype=dtype)

    if args.verbose:
        print_header(f'*** Initial Layer {args.layer_idx} ***')
        print(model_attn)
        print_header('*** Trainable Parameters ***')
        count = 0
        for n, p in model_attn.named_parameters():
            if p.requires_grad:
                print(f'├── {n} (requires_grad = {p.requires_grad}, dtype = {p.dtype})')
                count += 1
        if count == 0:
            print('(none)')

    # ---------------------------
    # Stage 1: Attention Transfer
    # ---------------------------
    if args.load_distill_checkpoint is None:
        dataloaders = load_data(data_dir, args.layer_idx, max_layer=num_hidden_layers, 
                                **distill_config.dataloader)
        train_loader = dataloaders['train']
        eval_loader  = dataloaders['validation']

        # Log some stats
        distill_config.model_train_params = count_parameters(model_attn, requires_grad=True)
        distill_config.model_total_params = count_parameters(model_attn, requires_grad=False)
        pct_trainable = distill_config.model_train_params / distill_config.model_total_params

        print_header('*** Distillation Parameter Counts ***')
        print(f'├── Number training to distill:  {distill_config.model_train_params}')
        print(f'├── Number of total parameters:  {distill_config.model_total_params}')
        print(f'├── Percent training to distill: {pct_trainable * 100:.3f}%')

        # Get optimizer and scheduler
        optimizer = get_optimizer(model=model_attn, **distill_config.optimizer)
        scheduler = get_scheduler(optimizer=optimizer, **distill_config.lr_scheduler)    
            
        # Load trainer 
        for arg, argv in distill_config.trainer.items():
            if arg != 'name':
                setattr(args, arg, argv)
        for _config in ['dataloader', 'optimizer', 'lr_scheduler']:
            setattr(args, _config, OmegaConf.to_container(getattr(distill_config, _config)))
            
        OurTrainer = get_trainer(distill_config.trainer.name)
        trainer = OurTrainer(model=model_attn, 
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
        model_attn.train_attention = True  # we did this above already
        model_attn = trainer.train()
        args.load_distill_checkpoint = trainer.best_val_checkpoint_path  # saved here
    else:
        with torch.no_grad():
            model_attn.load_state_dict(
                torch.load(args.load_distill_checkpoint)['model_state_dict'], strict=False,)

    # Prepare for 2nd stage finetune
    model_attn.train_attention = False
    if getattr(model_attn, 'base_attn', False):
        del model_attn.base_attn

    # --------------------------
    # Stage 2: Low-rank Adapting
    # --------------------------
    if args.max_finetune_steps is not None:
        args.max_steps = args.max_finetune_steps

    pretrained_model_config = pretrained_model_config.to_dict()  # Ordinarily not mutable, and 
                                                                 # TypeError: 'PretrainedConfig' object is not subscriptable
    pretrained_model_config['num_hidden_layers'] = 1  # only one layer
    pretrained_model_config = LlamaConfig.from_dict(pretrained_model_config)
    model = pretrained_model_class(pretrained_model_config)  # hacks
    with torch.no_grad():
        model.model.layers[0].self_attn = model_attn  # Should be the same
        model.model.layers[0].self_attn.load_state_dict(model_attn.state_dict())

    finetune_config, args = prepare_finetune_configs(args, model_config, args.finetune_config)
    dataloaders = load_data(data_dir, args.layer_idx, max_layer=num_hidden_layers, 
                            **distill_config.dataloader)
    train_loader = dataloaders['train']
    eval_loader  = dataloaders['validation']

    # Update distilling trainer to reflect layer-wise
    finetune_config.trainer.name = 'layer_finetune_xent_mse'
    
    checkpoint_path = args.load_finetune_checkpoint
    model, ft_peft_config = load_and_convert_finetune(model, finetune_config, 
                                                      checkpoint_path=checkpoint_path,  # could be None
                                                      print_model=False,  # args.verbose,
                                                      merge_loras=False,
                                                      peft_gradient_checkpointing=not args.no_peft_grad_ckpt,
                                                      add_self_attn_prefix=False,)
    model_attn = traverse_layers(model)[0].self_attn
    # Initialize optimizer and scheduler
    optimizer = get_optimizer(model=model_attn, **finetune_config.optimizer)
    scheduler = get_scheduler(optimizer=optimizer, **finetune_config.lr_scheduler)

    if args.verbose:
        print_header(f'*** Finetune Layer {args.layer_idx} ***')
        print(model_attn)
        print_header('*** Trainable Parameters ***')
        count = 0
        for n, p in model_attn.named_parameters():
            if p.requires_grad:
                print(f'├── {n} (requires_grad = {p.requires_grad}, dtype = {p.dtype})')
                count += 1
        if count == 0:  # no trainable parameters
            print('(none)')

        print_header(f'*** Teacher Layer {args.layer_idx} ***')
        print(teacher_attn)
        # assert teacher_attn.q_proj.weight == model_attn.q_proj.base_layer
    
    OurTrainer = get_trainer(finetune_config.trainer.name)
    for p in teacher_attn.parameters():
        p.requires_grad = False
    finetune_trainer = OurTrainer(model=model_attn, 
                                  teacher_layer=teacher_attn.to(model_attn.device),
                                  layer_idx=args.layer_idx,
                                  args=args,
                                  train_loader=train_loader,
                                  eval_loader=eval_loader,
                                  optimizer_and_scheduler=(optimizer, scheduler),
                                  device=args.device,
                                  wandb=wandb,
                                  checkpoint_suffix='_ft',
                                  save_results=False,
                                  **finetune_config.trainer)
    if args.verbose:
        print_header('Finetune config')
        print_config(finetune_config)
    print_header('*** Finetuning ***')
    print(f'├── Experiment name: {args.run_name}')
    print(f'├── Device: {args.device}')
    print(f'├── Seed: {args.seed}')
    model_attn = finetune_trainer.train()
    args.load_finetune_checkpoint = finetune_trainer.best_val_checkpoint_path

    if ft_peft_config is not None and wandb is not None:
        if not isinstance(ft_peft_config, dict):
            ft_peft_config = ft_peft_config.to_dict()
        _flattened['peft_ft'] = ft_peft_config
        wandb.config.update(_flattened, allow_val_change=True)  # saved here

    print_header('*** Done training ***')
    print('--> Saved Checkpoints:')
    print(f'--attn_mlp_checkpoint_path {args.load_distill_checkpoint} \\')
    print(f'--finetune_checkpoint_path {args.load_finetune_checkpoint} \\')

if __name__ == '__main__':
    main()
    print("Thanks for washing my dishes")
