import copy
import sys
sys.path.append('/home/simarora/code/lolcats/')

import os
from os.path import join
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import argparse
from omegaconf import OmegaConf
from tqdm import tqdm

import torch

# Modeling loading imports
from src.utils.setup import (
    seed_everything, update_config_from_args, update_model_config_from_args,
    flatten_config
)
from src.utils.logging import (
    print_header, print_config, _format_arg
)

from src.finetune import prepare_finetune_configs

from src.model.load_model import load_and_convert_attns, load_and_convert_finetune
from src.model.pretrained import get_pretrained_loader

# from llama_recipes.model_setup import update_model_config_from_args
# from llama_recipes.model_setup import toggle_attention, remove_base_attention
from llama_recipes.model_checkpointing.distill_checkpoint_handler import (
    load_sharded_model_single_gpu,
)
from llama_recipes.trainer_finetune import print_model_size


from src.model.load_model_for_eval import load_model_from_checkpoint, load_model_from_config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default='lolcats-eval')
    parser.add_argument("--model_type", type=str, default='lolcats_ckpt',
                        choices=['lolcats_ckpt', 'model_config', 'huggingface'])
    # Use these args to find sharded checkpoints
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--distill_config", type=str, default=None)
    parser.add_argument("--finetune_config", type=str, default=None)
    parser.add_argument("--eval_config", type=str, default=None)
    parser.add_argument("--config_dir", type=str, default='./configs')

    # Or just load the paths directly
    parser.add_argument("--attn_mlp_checkpoint_path", type=str, default=None)
    parser.add_argument("--finetune_checkpoint_path", type=str, default=None)

    # Override default configs
    parser.add_argument("--huggingface_token", type=str, default=None)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    # Feature map / model
    parser.add_argument("--attention_type", type=str, default=None)
    parser.add_argument("--lk_skip_connection", action='store_true', default=None)
    parser.add_argument("--lk_zero_init", action='store_true', default=None)
    parser.add_argument("--tie_qk_kernels", action='store_true', default=None)
    parser.add_argument("--train_qk", action='store_true', default=None)

    parser.add_argument("--dataset_chunk_size", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)

    ## Distributed training / Llama recipes
    parser.add_argument("--enable_fsdp", action='store_true', default=None)
    parser.add_argument("--low_cpu_fsdp", action='store_true', default=None)
    parser.add_argument("--pure_bf16", action='store_true', default=None)
    parser.add_argument("--fsdp_activation_checkpointing", action='store_true', default=None)
    parser.add_argument("--no_peft_grad_ckpt", action='store_true', default=None)

    # Miscellaneous
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_dir", type=str, default='./checkpoints')
    parser.add_argument("--verbose", action='store_true', default=False)
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--no_wandb", action='store_true', default=False)
    parser.add_argument("--wandb_entity", type=str, default='hazy-research')
    parser.add_argument("--replicate", type=int, default=None)
    
    args = parser.parse_args()

    distill_name = args.distill_config
    finetune_name = args.finetune_config
    
    # Run_name for loading checkpoints
    args.run_name = f'dl-d={distill_name}-m={args.model_config}-f={finetune_name}'
    if args.fsdp_activation_checkpointing is not None:
        args.run_name += f'-fac={args.fsdp_activation_checkpointing}'
    args.run_name = args.run_name.replace('True', '1').replace('False', '0')  # concise hacks

    return args


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
    config.save_model = True
    config.run_validation = True
    config.use_fp16 = False
    config.save_model = True
    config.save_optimizer = False
    config.output_dir = args.checkpoint_dir
    config.save_metrics = not args.no_wandb
    config.gradient_clipping = False
    config.gradient_clipping_threshold = 1.0
    config.num_epochs = getattr(config.trainer, 'num_train_epochs', None)
    config.num_train_steps = getattr(args, 'num_train_steps', None)  # exit training loop early for debugging
    config.eval_steps = getattr(config.trainer, 'eval_steps', None)  # how many gradient updates before evaluating
    return config


def check_state_dict_keys(_keys, layer_idx, rank=0):
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
    args = get_args()
    seed_everything(0)
    rank = 0

    args.checkpoint_dir = join(args.checkpoint_dir, args.model_config)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if args.model_type == 'lolcats_ckpt':
        distill_config_path = join('./configs/experiment', f'{args.distill_config}.yaml')
        distill_config = OmegaConf.load(distill_config_path)
        distill_config = update_config_from_args(distill_config, args)

    model_config_path = join('./configs/model', f'{args.model_config}.yaml')
    model_config = OmegaConf.load(model_config_path)
    model_config = update_model_config_from_args(model_config, args)

    # We load model initially onto CPU
    model_config.model.device_map = None  # FSDP will complain about device placement o.w.

    if args.model_type == 'lolcats_ckpt':
        # Update dataset pretrained model config
        for k in distill_config.dataset.pretrained_model_config:
            distill_config.dataset.pretrained_model_config[k] = getattr(model_config.model, k)

    args.run_name = args.run_name.replace('True', '1').replace('False', '0')  # concise hacks

    if args.model_type == 'lolcats_ckpt':
        distill_config = setup_fsdp_config(distill_config, args, 'distill')
        print_header('Distillation Config')
        print_config(distill_config)
    else:
        distill_config = OmegaConf.create({})
    print_header('Model Config')
    print_config(model_config)

    # Load the pre-trained model and setup its configuration
    model_loader = get_pretrained_loader(**model_config.model,
                                         huggingface_token=args.huggingface_token)
    model = model_loader.load(model_type='softmax')
    
    # convert
    model, distill_peft_config = load_and_convert_attns(model, model_config,
                                                            attention_type=args.attention_type,
                                                            checkpoint_path=None,
                                                            print_model=args.verbose,
                                                            merge_loras=False,
                                                            peft_gradient_checkpointing=not args.no_peft_grad_ckpt,
                                                            train_attention=False)
    model = load_sharded_model_single_gpu(model, model_path=args.attn_mlp_checkpoint_path, cfg=distill_config, rank=rank)
            
    finetune_config, args = prepare_finetune_configs(args, model_config,args.finetune_config)
    finetune_config = setup_fsdp_config(finetune_config, args, 'finetune')
    model, ft_peft_config = load_and_convert_finetune(model, finetune_config,
                                                        checkpoint_path=None,
                                                        print_model=args.verbose,
                                                        merge_loras=False,
                                                        peft_gradient_checkpointing=not args.no_peft_grad_ckpt)
    model = load_sharded_model_single_gpu(model, model_path=args.finetune_checkpoint_path, 
                                                    cfg=finetune_config, rank=rank)

    # save to .pt file
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