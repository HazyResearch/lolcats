"""
Alternative way to load trained models for evaluation
"""
import copy
import sys
from os.path import join
from omegaconf import OmegaConf

import torch

from src.utils.logging import print_header, print_config, _format_arg
from .pretrained import get_pretrained_loader
from .peft import create_peft_config
from .load_model import load_and_convert_attns
from .convert_model import remove_base_attention, toggle_attention

# Helpers
def get_args_from_checkpoint(fname: str):
    """
    Get arguments from checkpoint filename
    """
    id_to_name = {
        'lk': 'learned_kernel',
        'tqk': 'tie_qk_kernels',
        'tq': 'train_qk',
        'lzi': 'lk_zero_init',
        'lsc': 'lk_skip_connection',
        'pmnop': 'pretrained_model_name_or_path',
    }
    id_to_type = {
        'lk':  str,
        'tqk': bool,
        'tq':  bool,
        'lzi': bool,
        'lsc': bool,
        'pmnop': str,
    }
    args = {v: None for k, v in id_to_name.items()}
    args['run_name'] = ''
    
    for id_val in fname.split('-'):
        try:
            _id, val = id_val.split('=')
            if val[-len('_distill.pt'):] == '_distill.pt':  # hardcode hack
                val = val[:-len('_distill.pt')]
            if _id in id_to_type:
                _type = id_to_type[_id]
                args[id_to_name[_id]] = _type(val)
        except Exception:
            pass
    return OmegaConf.create(args)


def update_model_config_from_args(model_config, args):
    """Override default configs"""
    # Overall attention distillation
    for arg in ['learned_kernel', 'tie_qk_kernels', 'train_qk']:
        argval = getattr(args, arg)
        if argval is not None:
            setattr(model_config['attention'], arg, argval)
            args.run_name += f'-{_format_arg(arg)}={argval}'
    # Learned kernel
    for arg in ['lk_skip_connection', 'lk_zero_init']:
        argval = getattr(args, arg)
        if argval is not None:
            setattr(model_config['attention']['learned_kernel_kwargs'], 
                    arg[len('lk_'):], argval)
            args.run_name += f'-{_format_arg(arg)}={argval}'
    # Pretrained model
    if args.pretrained_model_name_or_path is not None:  # if specified 
        pmnop = args.pretrained_model_name_or_path
        model_config.model.pretrained_model_name_or_path = pmnop
        args.run_name += f'-pmnop={pmnop.split("/")[-1]}'
    return model_config



def get_lm_eval_model(model_kwargs: dict,  # model_loader.loading_kwargs
                      path_to_lm_eval_harness: str,  # ../../lm-evaluation-harness
                      hedgehog_model: bool = False,
                      long_model: bool = False,
                     ):
    """
    Load model for evaluation using LM Evaluation Harness
    """
    lm_kwargs = copy.deepcopy(model_kwargs)
    lm_kwargs['pretrained'] = lm_kwargs['pretrained_model_name_or_path']
    lm_kwargs['dtype'] = str(lm_kwargs['torch_dtype']).split('.')[-1]
    del lm_kwargs['torch_dtype']

    # lm_kwargs['use_cache'] = False
    lm_kwargs['output_attentions'] = False
    lm_kwargs['output_hidden_states'] = False

    print('-> Loading as lm-evaluation-harness model')
    if hedgehog_model:
        if 'mistral' in lm_kwargs['pretrained']:
            from lm_eval_harness.models import LolcatsMistralForCausalLM as ModelClass
        else:
            from lm_eval_harness.models import LolcatsLlamaForCausalLM as ModelClass
        lm = ModelClass.create_from_arg_string('', lm_kwargs)
    else:
        sys.path.append(path_to_lm_eval_harness)
        from lm_eval.models import get_model
        lm = get_model('hf-causal-experimental').create_from_arg_string('', lm_kwargs)
    return lm


def load_model_from_config(model_config_name: str,
                           config_dir: str = './configs',
                           lm_eval_model: bool = False,
                           path_to_lm_eval_harness: str = '/juice2/scr2/mzhang/projects/lm-evaluation-harness',
                          ):
    """
    Load model from a config file
    """
    # Load model configs
    model_config_path = join(config_dir, 'model', f'{model_config_name}.yaml')
    model_config = OmegaConf.load(model_config_path)

    model_loader = get_pretrained_loader(**model_config.model)
    tokenizer = model_loader.load_tokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    if lm_eval_model:  # Instantiate as lm_eval.base.LM object
        lm = get_lm_eval_model(model_loader.loading_kwargs, path_to_lm_eval_harness)
        model = lm.model
    else:
        model = model_loader.load()

    model.eval()
    if lm_eval_model: 
        lm.model = model
        model = lm
    return model, model_config, tokenizer                   


def load_model_from_checkpoint(attn_mlp_checkpoint_path: str = None,
                               finetune_checkpoint_path: str = None,
                               config_dir: str = './configs',
                               print_model: bool = False, 
                               debug: bool = False,
                               lm_eval_model: bool = False,
                               path_to_lm_eval_harness: str = '/juice2/scr2/mzhang/projects/lm-evaluation-harness',
                               profile_model: bool = False,
                              ):
    """
    Load model architecture from a checkpoint path
    -> attn_mlp_checkpoint_path should direct to checkpoint with learned MLPs
    -> finetune_checkpoint_path should direct to checkpoint with all other parameters
    -> Assumes checkpoint_path stings have names for model_config and finetune_configs
    """

    # Load model configs
    if attn_mlp_checkpoint_path is not None:
        if len(attn_mlp_checkpoint_path.split('/')) == 4:
            model_config = attn_mlp_checkpoint_path.split('/')[2]
        else:
            model_config = attn_mlp_checkpoint_path.split('/')[-1].split('-m=')[-1].split('-')[0]
        model_config_path = join(config_dir, 'model', f'{model_config}.yaml')
        model_config = OmegaConf.load(model_config_path)
        args = get_args_from_checkpoint(attn_mlp_checkpoint_path.split('/')[-1])
        model_config = update_model_config_from_args(model_config, args)
    else:
        if len(finetune_checkpoint_path.split('/')) == 4:
            model_config = finetune_checkpoint_path.split('/')[2]
        else:
            model_config = finetune_checkpoint_path.split('/')[-1].split('-m=')[-1].split('-')[0]
        model_config_path = join(config_dir, 'model', f'{model_config}.yaml')
        model_config = OmegaConf.load(model_config_path)

    if profile_model:
        model_config['attention']['attention_type'] += '_profile'

    if finetune_checkpoint_path is not None:
        finetune_config = finetune_checkpoint_path.split('-f=')[-1].split('-')[0]
        finetune_config_path = join(config_dir, 'experiment', f'{finetune_config}.yaml')
        finetune_config = OmegaConf.load(finetune_config_path)

    if debug:
        print_header('-- Model Config --')
        print_config(model_config)
        try:
            print_header('-- Finetune Config --')
            print_config(finetune_config)
        except NameError:
            pass
    
    # Get base model
    model_loader = get_pretrained_loader(**model_config.model)
    tokenizer = model_loader.load_tokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    if lm_eval_model and attn_mlp_checkpoint_path is not None:
        lm = get_lm_eval_model(model_loader.loading_kwargs, path_to_lm_eval_harness,
                               hedgehog_model=True)
        model = lm.model  # Do this way because we call the larger object
    elif lm_eval_model:   # Instantiate as lm_eval.base.LM object
        lm = get_lm_eval_model(model_loader.loading_kwargs, path_to_lm_eval_harness)
        model = lm.model
    elif attn_mlp_checkpoint_path is None:
        model = model_loader.load()
    else:
        model = model_loader.load(model_type=model_config['attention']['attention_type'])
    try:
        model.state_chunk_len = model_config['attention']['state_chunk_len']
    except KeyError:
        pass

    if attn_mlp_checkpoint_path is not None:
        # Update and load attentions
        model = load_and_convert_attns(model, model_config,
                                       checkpoint_path=attn_mlp_checkpoint_path)[0]
        if 'peft' in model_config['attention']:  # Merge back q and k proj
            model = model.merge_and_unload()
        # Already removed in load_and_convert_attns
        # model = remove_base_attention(model)  # , model_config.attention)
        model = toggle_attention(model, False)
        if debug:
            print_header('*** Model after attention converion ***')
            print(model)

    if finetune_checkpoint_path is not None:
        # Update architecture with LoRAs
        if finetune_config.finetune.method == 'lora':
            model, _ = create_peft_config(model, finetune_config.finetune)
        else:
            for p in model.parameters():
                p.requires_grad = True
    
        # Load weights
        state_dict = torch.load(finetune_checkpoint_path)['model_state_dict']
        _keys = model.load_state_dict(state_dict, strict=False)
        try:
            assert len(_keys.unexpected_keys) == 0
            print_header('*** All expected keys matched successfully ***')
        except AssertionError:
            print_header('*** Error: unexpected keys in checkpoint ***')
            print('Unexpected keys:')
            for k in _keys.unexpected_keys:
                print(k)
        if debug:
            print_header('Missing keys:')
            for k in _keys.missing_keys:
                print(k)
            print_header('Unexpected keys:')
            for k in _keys.unexpected_keys:
                print(k)

    try:
        # model = model.merge_and_unload()
        print('-> Training attention:', model.model.layers[0].self_attn.train_attention)
    except AttributeError as e:
        print('Error at:', e)
        _train_attn = model.model.model.layers[0].self_attn.train_attention
        print(f"But it's ok, {type(model.model.model)} has attribute 'layers'")
        print('-> Training attention:', _train_attn)
        

    if print_model or debug:  # Look at model
        print_header('*** Model ***')
        print(model)
        print_header('*** Trainable Parameters ***')
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f'├── {n}.requires_grad: {p.requires_grad}')   
    model.eval()
    if lm_eval_model:
        lm.model = model
        model = lm
    return model, model_config, tokenizer
