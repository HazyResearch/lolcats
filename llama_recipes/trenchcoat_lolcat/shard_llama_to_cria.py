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


# -----------
# Mini Llamas
# -----------
class LlamaMiniDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int, 
                 apply_input_layernorm: bool = True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.apply_input_layernorm = apply_input_layernorm  # Hack, but patch for saving attention inputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        # apply_input_layernorm: Optional[bool] = True,  # Ours
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states

        if self.apply_input_layernorm:  # Ours
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaMiniModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Identity() #nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            LlamaMiniDecoderLayer(config, layer_idx, apply_input_layernorm=layer_idx > 0) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class LlamaMiniModelForCausalLM(LlamaForCausalLM):
    """
    Pass in `inputs_embeds` for model.forward()
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaMiniModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Identity() #nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


def main():
    # ------
    # SET UP
    # ------
    args = get_args()
    checkpoint_dir = f"/data_ephemeral/sim/sharded_layers_405b_interval{args.layers_per_model}/"
    # checkpoint_dir = "/data_ephemeral/sim/sharded_layers_70b/"
    if args.enable_fsdp:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])

    # Where to save the output model checkpoints?
    checkpoint_dir = join(checkpoint_dir, args.model_config)
    if not os.path.isdir(checkpoint_dir) and ((args.enable_fsdp and rank == 0 and local_rank == 0) or not args.enable_fsdp):
        os.makedirs(checkpoint_dir)

        # Save individual .pt model weights in a subdirectory
        checkpoint_dir = join(checkpoint_dir, 'sharded_layers')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)

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
    

    # with torch.no_grad():
    with torch.device('meta'):
        mini_llama = LlamaMiniModelForCausalLM(mini_config).to(torch.bfloat16)
    mini_llama = mini_llama.to_empty(device='cpu')
    if rank == 0:
        print(mini_llama)
        print(f"Initialized mini llama.")
        print(model)
        print(model.state_dict().keys())
        # print(f"Done with the keys\n\n")
    
    mini_init = {}
    for i in range(args.layers_per_model): mini_init[i] = False
    with torch.no_grad():
        first = 0
        for layer_idx, layer in enumerate(tqdm(traverse_layers(model))):
            print(f'Saving layer attentions to {checkpoint_dir}...')
            
            pretrained_fname = model_config['model']['pretrained_model_name_or_path'].replace('/', '_')
            pretrained_fname = pretrained_fname.replace("_data_ephemeral_rahul_models_Meta-Llama-3.1-405B", "_scratch_rahul_models_Meta-Llama-3.1-405B")
            print(pretrained_fname)
            if layer_idx == 0 and rank == 0: 
                print(layer.state_dict().keys())

            
            mini_init[layer_idx % args.layers_per_model] = True
            mini_llama.model.layers[layer_idx % args.layers_per_model].load_state_dict(layer.state_dict()) # SA Flag
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