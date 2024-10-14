"""
Helpers to load checkpoints for learned feature maps (attentions) or other parameters
"""
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.utils.logging import print_header, _format_arg
from .convert_model import convert_attention
from .peft import create_peft_config
from src.model.utils import count_parameters


def load_and_convert_attns(model: nn.Module,
                           model_config: dict,
                           attention_type: str = None,
                           checkpoint_path: str = None,
                           print_model: bool = False,
                           merge_loras: bool = False,
                           train_converted: bool = True,  # Should be false if loading distill checkpoint by default
                           peft_gradient_checkpointing: bool = None,
                           train_attention: bool = False,  # Should be true if converting attentions for first time,
                           freeze_weights: bool = True,
                           rank: int = 0,
                           remove_base_attn: bool = True,
                          ) -> nn.Module:
    
    """
    Load trained attention kernel parameter weights
    """
    if freeze_weights:
        for p in model.parameters():
            p.requires_grad = False

    if attention_type is not None:  # override default
        model_config['attention']['attention_type'] = attention_type
    model_config['attention']['rank'] = rank   # multi-gpu debugging

    model = convert_attention(model, model_config['attention'], 
                              train_attention, remove_base_attn)

    # Add low-rank adapters
    peft_key = 'peft'  # inconsistency across configs... why do this to myself
    if 'peft_config' in model_config['attention']:
        peft_key = 'peft_config'
    if peft_key in model_config['attention']:  #  and not prior_loras:
        peft_config = model_config['attention'][peft_key]
        model, peft_config = create_peft_config(model, peft_config, 
                                                model_config['model']['torch_dtype'],
                                                preserve_requires_grad=train_converted,
                                                use_gradient_checkpointing=peft_gradient_checkpointing)
    else:
        peft_config = None

    if print_model and rank == 0:  # Look at model
        print_header('*** Model before checkpoint load ***')
        print(model)

    # Load any trained attentions
    if checkpoint_path is not None:
        print(f'Loading weights from {checkpoint_path}...')
        state_dict = torch.load(checkpoint_path)['model_state_dict']
        _keys = model.load_state_dict(state_dict, strict=False)
        try:
            assert len(_keys.unexpected_keys) == 0
            if rank == 0:
                print_header('*** All expected keys matched successfully ***')
                if print_model:
                    for k in state_dict.keys():
                        print(k)
        except Exception as e:
            if rank == 0:
                print(e)
                print_header('*** Error: unexpected keys in checkpoint ***')
                print('Unexpected keys:')
                for k in _keys.unexpected_keys:
                    print(k)
    if print_model and rank == 0:  # Look at model
        print_header('*** Model ***')
        print(model)
    if merge_loras:
        model = model.merge_and_unload()
        if print_model and rank == 0:
            print_header('*** Model (after merging adapters) ***')
            print(model)
    if print_model and rank == 0:  # Look at model
        print_header('*** Trainable Parameters ***')
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f'├── {n} (dtype = {p.dtype})')
        model_train_params = count_parameters(model, requires_grad=True)
        model_total_params = count_parameters(model, requires_grad=False)
        pct_trainable = model_train_params / model_total_params

        print_header('*** Distillation Parameter Counts ***')

        print(f'├── Number training to distill:  {model_train_params}')
        print(f'├── Number of total parameters:  {model_total_params}')
        print(f'├── Percent training to distill: {pct_trainable * 100:.3f}%')
    return model, peft_config


def load_and_convert_finetune(model: nn.Module,
                              finetune_config: dict,
                              checkpoint_path: str = None,
                              print_model: bool = False,
                              merge_loras: bool = False,
                              peft_gradient_checkpointing: bool = None,
                              rank: int = 0,
                              **peft_kwargs: any):
    """
    Load trained adapter / model weights
    """
    # Add low-rank adapters
    peft_config = None
    if finetune_config.finetune.method == 'lora':
        if getattr(finetune_config.finetune, 'kwargs', None) is not None:
            model, peft_config = create_peft_config(
                model, finetune_config.finetune,
                use_gradient_checkpointing=peft_gradient_checkpointing,
                **peft_kwargs,
            )
        # Keep specified weights trainable
        if 'trainable_weights' in finetune_config.finetune:
            for name in finetune_config.finetune['trainable_weights']:
                for n, p in model.named_parameters():
                    if name in n:
                        p.requires_grad = True
    else:
        for p in model.parameters(): #'model.layers.31.self_attn.feature_map_q.mlp.layer',
            p.requires_grad = True

    # Load weights
    if checkpoint_path:
        print(f"Loading weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path)['model_state_dict']
        _keys = model.load_state_dict(state_dict, strict=False)
        if len(_keys.unexpected_keys) > 0:
            new_state_dict = {k.replace('model.', 'model.model.'): v for k, v in state_dict.items()}
            _keys = model.load_state_dict(new_state_dict, strict=False)
        if len(_keys.unexpected_keys) > 0:
            new_state_dict = {k.replace('model.', 'base_model.model.model.'): v for k, v in state_dict.items()}
            _keys = model.load_state_dict(new_state_dict, strict=False)

        try:
            assert len(_keys.unexpected_keys) == 0
            if rank == 0:
                print_header('*** All expected keys matched successfully ***')
        except Exception as e:
            if rank == 0:
                print(e)
                print_header('*** Error: unexpected keys in checkpoint ***')
                print('Unexpected keys:')
                for k in _keys.unexpected_keys:
                    print(k)

    if print_model and rank == 0:  # Look at model
        print_header('*** Model ***')
        print(model)

    if merge_loras:
        try:
            model = model.merge_and_unload()
            if print_model and rank == 0:
                print_header('*** Model (after merging adapters) ***')
                print(model)
        except Exception as e:
            print(e)

    if print_model and rank == 0: 
        print_header('*** Trainable Parameters ***')
        count = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f'├── {n}.requires_grad: {p.requires_grad}')
                count += 1
        if count == 0:
            print('(none)')

    return model, peft_config
