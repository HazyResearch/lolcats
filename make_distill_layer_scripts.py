"""
Based on input args, create layer distill scripts

python make_distill_layer_scripts.py \
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


script = """python distill_llama_layer.py --layer_idx {layer_idx} --device {device}"""


def main():
    args = get_args()

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

    # Get model class and configs for layer instantiating
    pretrained_model_config = LlamaConfig.from_pretrained(model_config['model']['pretrained_model_name_or_path'])
    num_hidden_layers = pretrained_model_config.num_hidden_layers  # 32

    print('')
    print('-' * 10)
    for layer_idx in range(num_hidden_layers):
        _script = script.format(layer_idx=layer_idx, device=layer_idx)
        for _arg in dir(args):
            if _arg[0] != '_' and _arg != 'run_name':
                argval = getattr(args, _arg)
                if isinstance(argval, bool):
                    _script += f' --{_arg}'
                elif isinstance(argval, int) or isinstance(argval, str):
                    _script += f' --{_arg} {argval}'
        print(_script)
        print('')


if __name__ == "__main__":
    main()
        