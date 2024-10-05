""" 
This file just needs to save out the shards for 405B. 

Notes:
- Make sure that register_buffer inv_freq persistent=True for your modelling_llama.py 


/home/simarora/code/lolcats/checkpoints/dl-d=llama3_1_405b/rp_distill_llama_405b_xent1_mse1000_lr1e-2-m=llama3_1_405b/distill_llama3_1_405b_lk_smd_wtk64_fd64_w01-f=llama3_1_405b

python llama_recipes/stitch_mini_405b_sim2_rp_2048.py \
--model_config llama3_1_405b/distill_llama3_1_405b_lk_smd_wtk64_fd64_w01 \
--distill_config llama3_1_405b/distill_rpcontig2048_dcs2048_xent0_mse1000_lr1e-2 \
--finetune_config llama3_1_405b/rp_contig_finetune_llama_405b_qv_hparams_2048 \
--final_finetune_config llama3_1_405b/finetune_llama_405b_qkvo_e2_rp_2048 \
--verbose --replicate 0 --seed 0 \
--layers_per_model 3 --layer_idx 0 \
--enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing

"""

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
# from distill_llama import setup_fsdp_config
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
from distill_llama import (
    setup_wandb, get_args,  get_run_name_from_checkpoint,
    get_dataloaders, setup_fsdp_config
)


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default='lolcats')
    parser.add_argument("--layers_per_model", type=int)
    parser.add_argument("--layer_idx", type=int)  # specify starting layer
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--load_finetuned_loras", action='store_true', default=False)
    parser.add_argument("--no_distill", action='store_true', default=False)

    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--distill_config", type=str, default=None)
    parser.add_argument("--finetune_config", type=str, default=None)
    parser.add_argument("--eval_config", type=str, default=None)

    parser.add_argument("--final_finetune_config", type=str, default=None)

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
    parser.add_argument("--checkpoint_dir", type=str, default='./checkpoints')  # changed
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


def setup_fsdp_config(config, args, checkpoint_name: str = 'finetune', output_dir: str = None):
    """
    Hacky arguments for llama-recipes training function
    -> hardcoded save path
    """
    config.seed = args.seed
    config.enable_fsdp = args.enable_fsdp
    config.low_cpu_fsdp = args.low_cpu_fsdp
    config.dist_checkpoint_root_folder = args.checkpoint_dir  # '/home/mzhang/projects/lolcats/checkpoints/'
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
    config.num_epochs = config.trainer.num_train_epochs
    config.num_train_steps = getattr(args, 'num_train_steps', None)  # exit training loop early for debugging
    config.eval_steps = config.trainer.eval_steps  # how many gradient updates before evaluating
    return config


def main():
    # ------
    # SET UP
    # ------
    args = get_args()
    CHECKPOINT_DIR_405B = "/home/rahul/code/clean/lolcats/checkpoints"  #  "/data_ephemeral/sim/sharded_layers_405b/"
    CHECKPOINT_MODEL_CONFIG = 'llama3_1_405b/distill_llama3_1_405b_lk_smd_wtk64_fd64_w01'

    # Checkpoints are at:
    # /home/rahul/code/clean/lolcats/checkpointsdl-d=llama3_1_405b/rp_distill_llama_405b_xent1_mse1000_lr1e-2-m=llama3_1_405b/distill_llama3_1_405b_lk_smd_wtk64_fd64_w01-f=llama3_1_405b

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

    # for arg, argv in distill_config.trainer.items():  # legacy, should be removed
    #     if arg != 'name':
    #         setattr(args, arg, argv)
    #     for _config in ['dataloader', 'optimizer', 'lr_scheduler']:
    #         setattr(args, _config, OmegaConf.to_container(getattr(distill_config, _config)))

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
            # if args.final_finetune_config is not None:  # Update checkpoint for e2e finetune and lora loading
            #     kwargs['run_name'] += f'-ef={args.final_finetune_config.replace("llama3_1_405b/", "")}'
            # kwargs['run_name'] += f'-ft_lora={args.load_finetuned_loras}'.replace('True', '1').replace('False', '0')
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

    try:
        if not os.path.exists(model_config.model.pretrained_model_name_or_path):
            print(f"Model path {model_config.model.pretrained_model_name_or_path} does not exist. Using backup path. {model_config.model.pretrained_model_name_or_path_backup}")
            model_config.model.pretrained_model_name_or_path = model_config.model.pretrained_model_name_or_path_backup
        model_config.model.pop("pretrained_model_name_or_path_backup")
    except:
        print(f"Model without model.pretrained_model_name_or_path_backup path")
        pass

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
    # name_suffix = f'in={start:0{max_digits}d}-out={end:0{max_digits}d}'
    # args.run_name += f'-{name_suffix}'  # will save layer-wise checkpoints
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
        attention_type=None, # specified in model_config,
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
                # file name
                # /home/simarora/code/lolcats/checkpoints/dl-d=llama3_1_405b/rp_distill_llama_405b_xent1_mse1000_lr1e-2-m=llama3_1_405b/distill_llama3_1_405b_lk_smd_wtk64_fd64_w01-f=llama3_1_405b/rp_finetune_llama_40b_qv_hparams-s=0-se=0-re=0-in=000-out=008_distill.pt
                # args.distill_config = llama3_1_405b/rp_distill_llama_405b_xent1_mse1000_lr1e-2 
                # args.finetune_config = llama3_1_405b/rp_finetune_llama_40b_qv_hparams
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
            # args.run_name += f'-{name_suffix}'  # dealing with legacy naming

    
    # Step 4. Add finetuning parameters. 
    final_finetune_config, args = prepare_finetune_configs(args, model_config, 
                                                           args.final_finetune_config)
    final_finetune_config = setup_fsdp_config(final_finetune_config, args, 'finetune',
                                              output_dir='/home/mzhang/projects/lolcats/results/llama3_1_405b')  # hardcode

    args.finetune_lr = None 
    model, _ = load_and_convert_finetune(model, final_finetune_config,
                                         checkpoint_path=None,
                                         print_model=args.verbose,
                                         merge_loras=False,
                                         peft_gradient_checkpointing=not args.no_peft_grad_ckpt,
                                         rank=rank)
    

    # Step 5. Add the lora weights from mini-distill. 
    if args.load_finetuned_loras:
        print(f"Loading loras")
        with torch.no_grad():
            first = 0
            # args.run_name = finetune_layer_mini_xent1_mse1000-s=0-se=0-re=0-bs=1-gas=8-nte=2-ms=-1-se=0-re=0-in=000-out=008_ft.pt

            for layer_idx, layer in enumerate(tqdm(traverse_layers(model))):
                # example file names: 
                # ./checkpoints/ft-dl-d=0000_out=008_distill0d-m=llama3_1_405b/distill_llama3_1_405b_lk_smd_wtk64_fd64_w01-f=llama3_1_405b/finetune_layer_mini_xent1_mse1000-s=0-se=0-re=0-in=000-out=008-se=0-re=0_ft.pt
                # ./checkpoints/ft-dl-d=0001_out=125_distill1d-m=llama3_1_405b/distill_llama3_1_405b_lk_smd_wtk64_fd64_w01-f=llama3_1_405b/finetune_layer_mini_xent1_mse1000-s=0-se=0-re=0-in=117-out=125-se=0-re=0_ft.pt
                # 'finetune_layer_mini_xent1_mse1000-s=0-se=0-re=0-in=099-out=107_distill.pt'
                # 'finetune_layer_mini_xent1_mse1000-s=0-se=0-re=0-in=108-out=116_distill.pt'
                # 'finetune_layer_mini_xent1_mse1000-s=0-se=0-re=0-in=117-out=125_distill.pt'

                # ft-dl-d=0000_out=098_distill0d-m=llama3_1_405b  # matching based on assuming we loaded in the distill checkpoints last time
                # ft-dl-d=0000_out=107_distill1d-m=llama3_1_405b
                # ft-dl-d=0001_out=116_distill1d-m=llama3_1_405b
                
                # Get distill checkpoint name first
                # load_file_name = f'{join(args.checkpoint_dir, args.run_name)}'
                load_file_name = join(CHECKPOINT_DIR_405B, f'dl-d={args.distill_config}-m={CHECKPOINT_MODEL_CONFIG}-f={args.finetune_config}')
                load_file_name += f'-s={args.seed}-se={args.seed}-re={args.replicate}'
                max_digits = len(str(num_hidden_layers))
                start, end = first, first + (args.layers_per_model - 1)
                name_suffix = f'in={start:0{max_digits}d}-out={end:0{max_digits}d}'
                load_file_name += f'-{name_suffix}_distill.pt' 
                load_file_name = load_file_name.replace('True', '1').replace('False', '0')  # concise hacks
                args.load_distill_checkpoint = load_file_name
                print('args.load_distill_checkpoint:', args.load_distill_checkpoint)
                args.run_name = get_run_name_from_args(args)

                # load_file_name = f'distill_llama3_1_405b_lk_smd_wtk64_fd64_w01-f=llama3_1_405b/finetune_layer_mini_xent1_mse1000-s=0-se=0-re=0-{name_suffix}_distill.pt'  # hardcoded hack
                
                args.run_name = join(CHECKPOINT_DIR_405B, 'ft-' + args.run_name)
                args.run_name += f'-se={args.seed}-re={args.replicate}-{name_suffix}-se={args.seed}-re={args.replicate}_ft.pt'
                load_file_name = args.run_name.replace('True', '1').replace('False', '0')  # concise hacks

                if (layer_idx + 1) % args.layers_per_model == 0:
                    if rank == 0 or not args.enable_fsdp:
                        print(f'Loading layer loras from {CHECKPOINT_DIR_405B}...')
                        mini_weights = torch.load(load_file_name)['model_state_dict']
                        mini_weights = rename_state_dict(mini_weights, first)
                        _keys = model.load_state_dict(mini_weights, strict=False)
                        check_state_dict_keys(_keys, first)
                    first = layer_idx + 1
    
    # # Actual final run name / checkpoint naming
    # args.final_finetune_config = args.final_finetune_config.replace('llama3_1_405b/', '')
    # if args.final_finetune_config is not None:  # Update checkpoint for e2e finetune and lora loading
    #     args.run_name += f'-ef={args.final_finetune_config}'
    # args.run_name += f'-ft_lora={args.load_finetuned_loras}'.replace('True', '1').replace('False', '0')
    # if args.no_distill:
    #     args.run_name += '-no_distill'
    # final_finetune_config = setup_fsdp_config(final_finetune_config, args, 'finetune',
    #                                           output_dir='/home/mzhang/projects/lolcats/results/llama3_1_405b')  # hardcode


    
    hsdp_device_mesh = None
    if fsdp_config.hsdp and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
        hsdp_device_mesh = get_hsdp_device_mesh(replica_group_size=fsdp_config.replica_group_size,
                                                sharding_group_size=fsdp_config.sharding_group_size)
        print("HSDP device mesh is ready")


    # Step 6. Setup FSDP and load distilled checkpoints. And get some stuff (scheduler, optimizer, dataloader).
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

    if args.verbose and (rank == 0 or not args.enable_fsdp):
        print_header('*** FSDP MODEL ***')
        print(model)
        print_header('*** Trainable Parameters ***')
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f'├── {n} (dtype = {p.dtype})')

    # Initialize the optimizer and learning rate scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.finetune_lr if args.finetune_lr is not None else final_finetune_config.optimizer.lr,
        weight_decay=getattr(distill_config.optimizer, 'weight_decay', 0.),
    )
    scheduler = get_scheduler(optimizer=optimizer, **final_finetune_config.lr_scheduler)

    if args.verbose and rank == 0:
        print('-> Optimizer:', optimizer)
        print('-> Scheduler:', scheduler)

    # Get data
    train_dataloader, eval_dataloader, final_finetune_config = get_dataloaders(final_finetune_config, tokenizer)
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
            print_config(final_finetune_config)
    
    # Start the training process
    max_optimizer_steps = getattr(distill_config.optimizer, 'max_optimizer_steps', None)
    results, best_checkpoint_path = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        gradient_accumulation_steps=final_finetune_config.trainer.gradient_accumulation_steps, # train_config.gradient_accumulation_steps,
        train_config=final_finetune_config,  # train_config,
        fsdp_config=fsdp_config if args.enable_fsdp else None,
        local_rank=local_rank if args.enable_fsdp else None,
        rank=rank if args.enable_fsdp else None,
        # max_optimizer_steps,
        wandb_run=wandb_run,
    )
                
    # Save best model checkpoint as single .pt file
    if fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
        pass  # Model checkpoint already saved
    elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
        # Load sharded weights across GPUs into model
        ignore_param_rule = lambda n, p: (
            not p.requires_grad  # and 'feature_map' not in n or ('v_proj' in n or 'o_proj' in n)
        )
        load_model_sharded(model, rank, final_finetune_config, ignore_param_rule)
        if rank == 0:  # debugging
            print_header('** Sanity check model weights **')
            for n, p in model.named_parameters():
                if ('layers.0.' in n and 'base_attn' not in n and 
                    '.0.mlp.' not in n and '.block_sparse_moe' not in n):
                    print(f'-> {n}:\n', p)
        if rank == 0 or not args.enable_fsdp:
            with torch.no_grad():
                state_dict = model.state_dict()
                keys_to_keep = [key for key in state_dict.keys() if 'lora' in key or 'window_factors' in key or 'feature_map' in key]
                # keys_to_keep = [key for key in state_dict.keys() if 'lora' in key]
                new_state_dict = {key: state_dict[key] for key in keys_to_keep}
                if args.final_finetune_config is not None:  # Update checkpoint for e2e finetune and lora loading
                    args.final_finetune_config = args.final_finetune_config.replace('llama3_1_405b/', '')
                    args.run_name += f'-ef={args.final_finetune_config}'
                args.run_name += f'-ft_lora={args.load_finetuned_loras}'.replace('True', '1').replace('False', '0')
                if args.no_distill:
                    args.run_name += '-no_distill'
                torch.save(new_state_dict, f'ckpt_lora-{args.run_name}.pt')
                for k, v in new_state_dict.items():
                    print(k)

    if not args.enable_fsdp or rank==0:
        for k,v in results.items():
            print(f'Key: {k}, Value: {v}')
            if not args.no_wandb:
                wandb_run.summary[f'ft_{k}'] = v
        print('-> Find weights at:', best_checkpoint_path)
        print('-> Find zipped checkpoint at:', f'ckpt_lora-{args.run_name}.pt')
        

if __name__ == '__main__':
    main()
    print("Thanks for washing my dishes")

