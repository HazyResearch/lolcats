"""
Alternate way to do things where we convert a block of Llama decoder layers into linear attention equivalents

This lets us linearize big models in a decentralized manner without interconnect. 
Just take a layer and train.

(screen -r h3)
python distill_llama_mini.py \
--model_config distill_llama3_8b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean_mini_xent1_mse1000 \
--lk_zero_init --lr 1e-3 \
--layer_idx 0 --layers_per_model 8 --device 0 \
--verbose --seed 0 --replicate 0 

(screen -r h4)
python distill_llama_mini.py \
--model_config distill_llama3_8b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean_mini_xent1_mse1000 \
--lk_zero_init --lr 1e-3 \
--layer_idx 8 --layers_per_model 8 --device 0 \
--verbose --seed 0 --replicate 0


python distill_llama_mini.py \
--model_config distill_llama3_8b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean_mini_xent1_mse1000 \
--lk_zero_init --lr 1e-3 \
--layer_idx 16 --layers_per_model 8 --device 0 \
--verbose --seed 0 --replicate 0

(screen -r h3)
python distill_llama_mini.py \
--model_config distill_llama3_8b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean_mini_xent1_mse1000 \
--lk_zero_init --lr 1e-3 \
--layer_idx 24 --layers_per_model 8 --device 0 \
--verbose --seed 0 --replicate 0


python distill_llama_mini.py \
--model_config distill_llama3_8b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean_mini_xent1_mse1000 \
--lk_zero_init --lr 1e-3 \
--layer_idx 0 --layers_per_model 8 --device 0 \
--verbose --seed 0 --replicate 0 
--layer_idx 16
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

from transformers import PretrainedConfig, LlamaConfig
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaAttention, LlamaMLP, LlamaRMSNorm, LlamaConfig, 
    LlamaModel, LlamaForCausalLM, LlamaRotaryEmbedding
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.utils.setup import (
    init_wandb, seed_everything, flatten_config, get_run_name_from_args,
    update_config_from_args, update_model_config_from_args,
)
from src.utils.logging import print_config, print_header
# from src.dataloaders import load_data
from src.trainer import get_trainer, get_optimizer, get_scheduler
from src.finetune import prepare_finetune_configs, get_finetuner

from src.model.pretrained import get_pretrained_loader
from src.model.load_model import load_and_convert_attns, load_and_convert_finetune
from src.model.convert_model import toggle_attention, remove_base_attention, traverse_layers
from src.model.utils import count_parameters
from src.model.convert_model import get_attention


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
        # MZ todo: explore things that'd involve non [seq_len] pos_ids
        return {'inputs_embeds': x}  # , 'position_ids': position_ids}


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

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
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
        self.lm_head = None  # nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        return CausalLMOutputWithPast(
            loss=None,
            logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


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
    start, end = args.layer_idx, args.layer_idx + args.layers_per_model - 1
    name_suffix = f'in={start:0{max_digits}d}-out={end:0{max_digits}d}'
    args.run_name += f'-{name_suffix}'  # will save layer-wise checkpoints
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

    mini_config = copy.deepcopy(pretrained_model_config).to_dict()
    mini_config['num_hidden_layers'] = args.layers_per_model
    mini_config['attn_implementation'] = 'eager'
    mini_config = LlamaConfig.from_dict(mini_config)
    model_config.model.attn_implementation = 'eager'
    
    try:  # Load relevant model weights from memory
        mini_llama = LlamaMiniModelForCausalLM(mini_config)
        with torch.no_grad():
            pretrained_fname = model_config['model']['pretrained_model_name_or_path'].replace('/', '_')
            pretrained_fname = join(args.checkpoint_dir, pretrained_fname) + f'-{name_suffix}.pt'
            mini_llama.load_state_dict(torch.load(pretrained_fname))
            print_header('All teacher weights loaded successfully')
        for p in mini_llama.parameters():   # Freeze all layers
            p.requires_grad = False

        mini_llama = load_and_convert_attns(mini_llama, model_config,
                                            attention_type=None, # specified in model_config,
                                            checkpoint_path=None,
                                            print_model=args.verbose,
                                            train_attention=True)[0]
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
        
        # Save pretrained Transformer weights
        mini_llama = LlamaMiniModelForCausalLM(mini_config)

        with torch.no_grad():
            first = 0
            for layer_idx, layer in enumerate(tqdm(traverse_layers(model), desc=f'Saving layer attentions to {args.checkpoint_dir}...')):
                pretrained_fname = model_config['model']['pretrained_model_name_or_path'].replace('/', '_')

                mini_llama.model.layers[layer_idx % args.layers_per_model].load_state_dict(layer.state_dict())
                if (layer_idx + 1) % args.layers_per_model == 0:
                    pretrained_fname = (
                        join(args.checkpoint_dir, pretrained_fname) + 
                        f'-in={first:0{max_digits}d}-out={layer_idx:0{max_digits}d}.pt'
                    )  # name_suffix = f'in={start:0{max_digits}d}-out={end:0{max_digits}d}'
                    torch.save(mini_llama.state_dict(), pretrained_fname)
                    first = layer_idx + 1
                    del mini_llama
                    mini_llama = LlamaMiniModelForCausalLM(mini_config)
        del model

        # Load relevant model weights
        mini_llama = LlamaMiniModelForCausalLM(mini_config)
        start, end = args.layer_idx, args.layer_idx + args.layers_per_model
        with torch.no_grad():
            pretrained_fname = model_config['model']['pretrained_model_name_or_path'].replace('/', '_')
            pretrained_fname = join(args.checkpoint_dir, pretrained_fname) + f'-{name_suffix}.pt'
            mini_llama.load_state_dict(torch.load(pretrained_fname))
            print_header('All teacher weights loaded successfully')
        for p in mini_llama.parameters():   # Freeze all layers
            p.requires_grad = False

        mini_llama = load_and_convert_attns(mini_llama, model_config,
                                            attention_type=None, # specified in model_config,
                                            checkpoint_path=None,
                                            print_model=args.verbose,
                                            train_attention=True)[0]

    device = torch.device(f'cuda:{args.device}')
    mini_llama = mini_llama.to(device, dtype=dtype)
    mini_llama.to(device)

    if args.verbose:
        print_header(f'*** Initial Layer {args.layer_idx} ***')
        print(mini_llama)
        print_header('*** Trainable Parameters ***')
        count = 0
        for n, p in mini_llama.named_parameters():
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
        distill_config.model_train_params = count_parameters(mini_llama, requires_grad=True)
        distill_config.model_total_params = count_parameters(mini_llama, requires_grad=False)
        pct_trainable = distill_config.model_train_params / distill_config.model_total_params

        print_header('*** Distillation Parameter Counts ***')
        print(f'├── Number training to distill:  {distill_config.model_train_params}')
        print(f'├── Number of total parameters:  {distill_config.model_total_params}')
        print(f'├── Percent training to distill: {pct_trainable * 100:.3f}%')

        # Get optimizer and scheduler
        optimizer = get_optimizer(model=mini_llama, **distill_config.optimizer)
        scheduler = get_scheduler(optimizer=optimizer, **distill_config.lr_scheduler)    
            
        # Load trainer 
        for arg, argv in distill_config.trainer.items():
            if arg != 'name':
                setattr(args, arg, argv)
        for _config in ['dataloader', 'optimizer', 'lr_scheduler']:
            setattr(args, _config, OmegaConf.to_container(getattr(distill_config, _config)))
            
        OurTrainer = get_trainer(distill_config.trainer.name)
        trainer = OurTrainer(model=mini_llama,
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
        mini_llama = toggle_attention(mini_llama, train=True)
        mini_llama = trainer.train()
        args.load_distill_checkpoint = trainer.best_val_checkpoint_path  # saved here
    else:
        with torch.no_grad():
            mini_llama.load_state_dict(
                torch.load(args.load_distill_checkpoint)['model_state_dict'], strict=False,)

    # Prepare for 2nd stage finetune
    # mini_llama = toggle_attention(mini_llama, train=False)  # keep this
    mini_llama = remove_base_attention(mini_llama)

    # --------------------------
    # Stage 2: Low-rank Adapting
    # --------------------------
    if args.max_finetune_steps is not None:
        args.max_steps = args.max_finetune_steps

    finetune_config, args = prepare_finetune_configs(args, model_config, args.finetune_config)
    try:
        train_loader = dataloaders['train']
        eval_loader  = dataloaders['validation']
    except:
        dataloaders = load_data(data_dir, args.layer_idx, max_layer=num_hidden_layers, 
                                **distill_config.dataloader)
        train_loader = dataloaders['train']
        eval_loader  = dataloaders['validation']
    
    checkpoint_path = args.load_finetune_checkpoint
    mini_llama, ft_peft_config = load_and_convert_finetune(mini_llama, finetune_config, 
                                                           checkpoint_path=checkpoint_path,  # could be None
                                                           print_model=False,  # args.verbose,
                                                           merge_loras=False,
                                                           peft_gradient_checkpointing=not args.no_peft_grad_ckpt,
                                                           add_self_attn_prefix=False,)
    # Initialize optimizer and scheduler
    optimizer = get_optimizer(model=mini_llama, **finetune_config.optimizer)
    scheduler = get_scheduler(optimizer=optimizer, **finetune_config.lr_scheduler)

    if args.verbose:
        print_header(f'*** Finetuning Layers {args.layer_idx} - {args.layer_idx + args.layers_per_model - 1} ***')
        print(mini_llama)
        print_header('*** Trainable Parameters ***')
        count = 0
        for n, p in mini_llama.named_parameters():
            if p.requires_grad:
                print(f'├── {n} (requires_grad = {p.requires_grad}, dtype = {p.dtype})')
                count += 1
        if count == 0:  # no trainable parameters
            print('(none)')

        # print_header(f'*** Teacher Layers {args.layer_idx} - {args.layer_idx + args.layers_per_model - 1} ***')
        # print(teacher_mini_llama)
        # assert teacher_attn.q_proj.weight == model_attn.q_proj.base_layer
    
    OurTrainer = get_trainer(finetune_config.trainer.name)
    finetune_trainer = OurTrainer(model=mini_llama,
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
    mini_llama = finetune_trainer.train()
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

