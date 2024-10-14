"""
Evaluate models with lm-evaluation-harness
- Another way to evaluate models that require multiple GPUs to load
- Right now does a heinous manual pipelining of model layers across devices
"""
import copy
import sys
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

# LM evaluation imports
# from lm_eval_harness.model_loader import load_model_from_checkpoint, load_model_from_config
from src.model.load_model_for_eval import load_model_from_checkpoint, load_model_from_config

LM_EVALUATION_HARNESS_PATH = '/home/mzhang/projects/lm-evaluation-harness'


OPEN_LLM = [  # task, shots
    ('arc-challenge', 25),
    ('hellaswag', 10),
    ('truthfulqa-mc', 0),
    ('winogrande', 5),
    ('gsm8k', 5),
]
ZERO_SHOT = [
    ('hellaswag', 0),
    ('piqa', 0),
    ('arc-challenge', 0),
    ('arc-easy', 0),
    ('winogrande', 0),
    ('hendrycksTest', 5),
]


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

    # LM Evaluation tasks
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--num_shots", type=int, default=0)

    # LM Evaluation Harness args
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_batch_size", type=int, default=None)
    parser.add_argument("--no_cache", action='store_true', default=None)
    parser.add_argument("--limit", type=float, default=None,  # helpful for debugging
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
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

    # Run name for evaluation
    args.experiment_name = f'd={args.task}-ns={args.num_shots}-d={distill_name}-m={args.model_config}-f={finetune_name}'
    if args.replicate is not None:
        args.experiment_name += f"-r={args.replicate}"
    args.experiment_name = args.experiment_name.replace('True', '1').replace('False', '0')  # concise hacks
    return args


def init_wandb(args):
    if args.no_wandb:
        wandb = None
    else:
        import wandb
        wandb.init(config={},
                   entity=args.wandb_entity,
                   name=args.experiment_name,
                   project=args.project_name)
    return wandb


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


def get_lm_eval_lolcats_model(model_kwargs: dict, lolcats_model: bool = True):
    lm_kwargs = copy.deepcopy(model_kwargs)
    lm_kwargs['pretrained'] = lm_kwargs['pretrained_model_name_or_path']
    lm_kwargs['dtype'] = str(lm_kwargs['torch_dtype']).split('.')[-1]
    del lm_kwargs['torch_dtype']

    print('-> Loading as lm-evaluation-harness model')
    if 'Llama' in lm_kwargs['pretrained_model_name_or_path']:  #  and lolcats_model:
        lm_kwargs['device_map'] = None
        from lm_eval_harness.models import ShardedLolcatsLlamaForCausalLM
        lm = ShardedLolcatsLlamaForCausalLM.create_from_arg_string(
            '', lm_kwargs,
        )
    else:
        sys.path.append(LM_EVALUATION_HARNESS_PATH)
        from lm_eval.models import get_model
        
        lm = get_model('hf-causal-experimental').create_from_arg_string(
            '', lm_kwargs,
        )
    return lm


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


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
    sys.path.append(LM_EVALUATION_HARNESS_PATH)
    from lm_eval import evaluator
    
    args = get_args()
    seed_everything(args.seed)
    rank = 0

    args.checkpoint_dir = join(args.checkpoint_dir, args.model_config)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if args.model_type == 'lolcats_ckpt':
        # Load distillation + lolcats attention configs
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
        # Use for loading sharded weights into single model
        distill_config = setup_fsdp_config(distill_config, args, 'distill')

        print_header('Distillation Config')
        print_config(distill_config)
    else:
        distill_config = OmegaConf.create({})
    print_header('Model Config')
    print_config(model_config)

    wandb = init_wandb(args)
    if wandb is not None:
        distill_config['model'] = model_config  # Combine for logging
        _flattened = {'model': model_config,
                      'model_config': args.model_config,  # config file names
                      'distill_config': args.distill_config,
                      'finetune_config': args.finetune_config,
                      'eval_config': args.eval_config,
                      'replicate': args.replicate,
                      # LM Eval args
                      'task': args.task,
                      'num_shots': args.num_shots,
                      'batch_size': 1 if args.batch_size is None else args.batch_size,
                      'max_batch_size': 1 if args.max_batch_size is None else args.max_batch_size,}
        flatten_config(OmegaConf.to_container(distill_config), _flattened, '')
        wandb.config.update(_flattened)

    # ------------------------
    # 2. LOAD PRETRAINED MODEL
    # ------------------------
    # Load the pre-trained model and setup its configuration
    # Initialize tokenizer and model loader
    model_loader = get_pretrained_loader(**model_config.model,
                                         huggingface_token=args.huggingface_token)
    tokenizer = model_loader.load_tokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    # Load model weights to CPU
    lm = get_lm_eval_lolcats_model(model_loader.loading_kwargs,
                                   lolcats_model=args.model_type == 'lolcats_ckpt')
    model = lm.model  # Do this way because we call the larger object
    for ix in range(len(model.model.layers)): 
        model.model.layers[ix].cpu()
    torch.cuda.empty_cache()
    for n, p in model.named_parameters():
        print(n, p.device)
    
    # Pipeline model manually across devices
    world_size = torch.cuda.device_count()
    print('Number of devices:', world_size)

    print('model.model.embed_tokens', f'{count_params(model.model.embed_tokens) / 1e9:.3f}B params')
    for ix in range(len(model.model.layers)):
        print(f'model.model.layers[{ix}]', f'{count_params(model.model.layers[ix]) /1e9:.3f}B params')
    print('model.model.norm', f'{count_params(model.model.norm) / 1e9:.3f}B params')
    print('model.lm_head', f'{count_params(model.lm_head) / 1e9:.3f}B params')

    # model.device = torch.device('cuda:0')
    print_header('*** Pipelining layers ***')
    model.model.embed_tokens.to(torch.device('cuda:0'))
    model.model.norm.to(torch.device('cuda:0'))
    model.lm_head.to(torch.device('cuda:0'))

    for ix in tqdm(range(len(model.model.layers)), colour='#643b9f'):
        device_id = ix % (world_size - 1)
        model.model.layers[ix].to(torch.device(f'cuda:{device_id + 1}'))
    
    print_header('Pretrained Model')
    print(model)
    
    model_config.model_name = model_config.model.pretrained_model_name_or_path
    print_model_size(model, model_config, 0)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if args.pure_bf16:
        model.to(torch.bfloat16)

    if args.model_type == 'lolcats_ckpt':
        # -------------------------------
        # 3. CONVERT DISTILLED ATTENTIONS
        # -------------------------------
        model, distill_peft_config = load_and_convert_attns(model, model_config,
                                                            attention_type=args.attention_type,  # 'lolcats_llama',
                                                            checkpoint_path=None,
                                                            print_model=args.verbose,
                                                            merge_loras=False,
                                                            peft_gradient_checkpointing=not args.no_peft_grad_ckpt,
                                                            train_attention=False)
        if True:  # rank == 0:
            if distill_config.trainer.name is not None and args.attn_mlp_checkpoint_path is not None:
                model = load_sharded_model_single_gpu(model, model_path=args.attn_mlp_checkpoint_path,  #  None,
                                                    cfg=distill_config, rank=rank)
            
        # ----------------------------
        # 4. ADD FINETUNING PARAMETERS
        # ----------------------------
        finetune_config, args = prepare_finetune_configs(args, model_config, 
                                                        args.finetune_config)
        finetune_config = setup_fsdp_config(finetune_config, args, 'finetune')
                
        model, ft_peft_config = load_and_convert_finetune(model, finetune_config,
                                                        checkpoint_path=None,
                                                        print_model=args.verbose,
                                                        merge_loras=False,
                                                        peft_gradient_checkpointing=not args.no_peft_grad_ckpt)
        if True:  # rank == 0:
            if '.pt' in args.finetune_checkpoint_path:
                with torch.no_grad():
                    _keys = model.load_state_dict(torch.load(args.finetune_checkpoint_path), strict=False)
                    check_state_dict_keys(_keys, 0)
            else:
                model = load_sharded_model_single_gpu(model, model_path=args.finetune_checkpoint_path,  #  None,
                                                    cfg=finetune_config, rank=rank)
            
        if True:  # if rank == 0:
            print_header('** Sanity check model weights **')
            for n, p in model.named_parameters():
                if 'layers.0.' in n:
                    print(f'-> {n}:\n', p)
    # Back to LM Eval model
    lm.model = model
    model = lm

    if args.task in ['mmlu', 'hendrycksTest']:
        from lm_eval.tasks import TASK_REGISTRY
        tasks = sorted([k for k in TASK_REGISTRY.keys() if f'{args.task}-' in k])
    else:
        tasks = [args.task]

    results = evaluator.simple_evaluate(
        model=model,
        model_args='',
        tasks=tasks,
        num_fewshot=args.num_shots,
        batch_size=1 if args.batch_size is None else args.batch_size,
        max_batch_size=1 if args.max_batch_size is None else args.max_batch_size,
        device=torch.device('cuda:0'),  # model.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict={},  # description_dict,
        decontamination_ngrams_path=None,  # args.decontamination_ngrams_path,
        check_integrity=None,  # args.check_integrity,
        write_out=False,  # args.write_out,
        output_base_path=None,  # args.output_base_path,
    )
    try:
        del results['config']['device']
    except:
        pass
    if args.task in ['mmlu', 'hendrycksTest', 'mmlu_cloze']:
        mmlu_accs = []
        for k, v in results['results'].items():
            if args.task in k:
                mmlu_accs.append(v['acc'])
        print(mmlu_accs)
        if len(mmlu_accs) > 0:
            results['results']['mmlu'] = {'acc': sum(mmlu_accs) / len(mmlu_accs)}
    
        print('MMLU RESULT:', results['results']['mmlu'])
    print(results)
    if wandb is not None:
        wandb.log(results)


if __name__ == '__main__':
    main()
