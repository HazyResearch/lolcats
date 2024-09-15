"""
Quick demo of linearized LLM generations

Example scripts:
```
python mmlu_demo_lolcats.py \
--attn_mlp_checkpoint_path ./checkpoints/distill_llama3_8b_lk_smd_wtk64_fd64_w01/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2-m=distill_llama3_8b_lk_smd_wtk64_fd64_w01-f=finetune_lora_qkvo_alpaca_clean-s=0-se=0-re=12-lzi=1_distill.pt \
--finetune_checkpoint_path ./checkpoints/distill_llama3_8b_lk_smd_wtk64_fd64_w01/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2-m=distill_llama3_8b_lk_smd_wtk64_fd64_w01-f=finetune_lora_qkvo_alpaca_clean-s=0-se=0-re=12-lzi=1-bs=1-gas=8-nte=2-se=0-re=12_ft.pt \
--num_shots 5 --split test --seed 0 --num_generations 5 --max_new_tokens 1

python mmlu_demo_lolcats.py \
--attn_mlp_checkpoint_path ./checkpoints/distill_long_llama3_8b_lk_smd_wtk64_fd64_w01/dl-d=distill_long_alpaca_8k_xent0_mse1000_lr1e-2_bs1-m=distill_long_llama3_8b_lk_smd_wtk64_fd64_w01-f=finetune_long_lora_qkvo_alpaca_clean_8192_bs1-s=0-nte=2-se=0-re=800-scl=1024-ws=64-lzi=1_distill.pt \
--finetune_checkpoint_path ./checkpoints/distill_long_llama3_8b_lk_smd_wtk64_fd64_w01/dl-d=distill_long_alpaca_8k_xent0_mse1000_lr1e-2_bs1-m=distill_long_llama3_8b_lk_smd_wtk64_fd64_w01-f=finetune_long_lora_qkvo_alpaca_clean_8192_bs1-s=0-nte=2-se=0-re=800-scl=1024-ws=64-lzi=1-bs=1-gas=1-nte=2-ms=-1-se=0-re=800_ft.pt \
--num_shots 5 --split test --seed 0 --num_generations 5 --max_new_tokens 1

python mmlu_demo_lolcats.py \
--attn_mlp_checkpoint_path ./checkpoints/distill_llama3_8b_lk_smd_wtk64_fd64_wo/dl-d=no_distill_alpaca_clean-m=distill_llama3_8b_lk_smd_wtk64_fd64_wo-f=finetune_lora_qkvo_alpaca_clean-s=0-nte=2-se=0-re=800-scl=1024-ws=64-lzi=1-nte=2-se=0-re=800_ft.pt \
--finetune_checkpoint_path ./checkpoints/distill_llama3_8b_lk_smd_wtk64_fd64_wo/dl-d=no_distill_alpaca_clean-m=distill_llama3_8b_lk_smd_wtk64_fd64_wo-f=finetune_lora_qkvo_alpaca_clean-s=0-nte=2-se=0-re=800-scl=1024-ws=64-lzi=1-nte=2-se=0-re=800_ft.pt \
--num_shots 5 --split test --seed 0 --num_generations 5 --max_new_tokens 1


python mmlu_demo_lolcats.py \
--attn_mlp_checkpoint_path ./checkpoints/distill_long_llama3_8b_lk_smd_wtk64_fd64_wo/dl-d=no_distill_alpaca_clean-m=distill_long_llama3_8b_lk_smd_wtk64_fd64_wo-f=finetune_long_lora_qkvo_alpaca_clean_8192_bs1-s=0-nte=2-se=0-re=800-scl=1024-ws=64-lzi=1-nte=2-se=0-re=800_ft.pt \
--finetune_checkpoint_path ./checkpoints/distill_long_llama3_8b_lk_smd_wtk64_fd64_wo/dl-d=no_distill_alpaca_clean-m=distill_long_llama3_8b_lk_smd_wtk64_fd64_wo-f=finetune_long_lora_qkvo_alpaca_clean_8192_bs1-s=0-nte=2-se=0-re=800-scl=1024-ws=64-lzi=1-nte=2-se=0-re=800_ft.pt \
--num_shots 5 --split test --seed 0 --num_generations 5 --max_new_tokens 1
```
"""
from typing import Optional, List
from os.path import join
import time
import argparse
import numpy as np
import torch

from tqdm import tqdm
from omegaconf import OmegaConf

from transformers import TextStreamer, TextIteratorStreamer, AutoTokenizer

from src.utils.setup import seed_everything
from src.utils.logging import print_header
from src.model.pretrained import get_pretrained_loader
from src.model.load_model import load_and_convert_attns, load_and_convert_finetune


from llama_recipes.model_checkpointing.distill_checkpoint_handler import (
    load_sharded_model_single_gpu,
)

from datasets import load_dataset

CACHE_DIR = '/scr-ssd/mzhang/data/mmlu'


class MMLU():
    def __init__(self, num_shots: int, cache_dir: str, split: str = 'dev'):
        self.num_shots = num_shots
        self.cache_dir = cache_dir
        self.split = split 

    def _format_subject(self, subject):
        words = subject.split("_")
        return " ".join(words)

    def get_description(self, subject):
        description = f"The following are multiple choice questions (with answers) about {self._format_subject(subject)}."
        return description

    def format_example(self, doc, keys):
        """
        <prompt>
        A. <choice1>
        B. <choice2>
        C. <choice3>
        D. <choice4>
        Answer:
        """
        question = doc["question"].strip()
        choices = "".join(
            [f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])]
        )
        prompt = f"{question}\n{choices}Answer:"
        answer = keys[doc['answer']]
        return prompt, answer

    def load_prompts(self):
        ds = load_dataset("cais/mmlu", "all", cache_dir=self.cache_dir)
        ds = ds[self.split]
        subjects = np.unique(ds['subject'])
        all_samples = []
        keys = ["A", "B", "C", "D"]
        for subject in tqdm(subjects, desc='processing subjects'):
            samples = [x for x in ds if x['subject'] == subject]
            # breakpoint()
            # Just get 1 sample for each subject
            ix = 0
            if len(samples) > self.num_shots + 1:  # number in context + final question
                prompt = self.get_description(subject)
                prompt += '\n\n'
                for _ in range(self.num_shots):
                    question, answer = self.format_example(samples[ix], keys)
                    prompt += f'{question} {answer}\n\n'
                    ix += 1
                question, answer = self.format_example(samples[ix], keys)
                prompt += f'{question}'
                all_samples.append((prompt, answer))
        return all_samples
    

def get_args():
    parser = argparse.ArgumentParser()
    # Model load + setup
    parser.add_argument("--attn_mlp_checkpoint_path", type=str, default=None)
    parser.add_argument("--finetune_checkpoint_path", type=str, default=None)
    parser.add_argument("--config_dir", type=str, default='configs')
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_shots", type=int, default=5)
    parser.add_argument("--split", type=str, default='test')

    # Generation
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    # Miscellaneous
    parser.add_argument("--benchmark", action='store_true', default=False)
    parser.add_argument("--print_model", action='store_true', default=False)
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--huggingface_token", type=str, default=None)

    # Alt
    parser.add_argument("--attn_checkpoint_path", type=str, default=None)
    parser.add_argument("--peft_checkpoint_path", type=str, default=None)

    args = parser.parse_args()
    if args.attn_mlp_checkpoint_path is None and args.attn_checkpoint_path is not None:
        args.attn_mlp_checkpoint_path = args.attn_checkpoint_path
    if args.finetune_checkpoint_path is None and args.peft_checkpoint_path is not None:
        args.finetune_checkpoint_path = args.peft_checkpoint_path
    return args


def get_lm_eval_lolcats_model(model_kwargs: dict, lolcats_model: bool = True):
    lm_kwargs = copy.deepcopy(model_kwargs)
    lm_kwargs['pretrained'] = lm_kwargs['pretrained_model_name_or_path']
    lm_kwargs['dtype'] = str(lm_kwargs['torch_dtype']).split('.')[-1]
    del lm_kwargs['torch_dtype']

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
    # model = lm.model
    return lm


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


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


def load_model_from_checkpoint(attn_mlp_checkpoint_path: str, 
                               finetune_checkpoint_path: str, 
                               config_dir: str = 'configs',
                               print_model: bool = False, 
                               debug: bool = False,
                               huggingface_token: str = None):
    rank = 0
    # Get configs from checkpoint paths
    try:
        model_config = attn_mlp_checkpoint_path.split('-m=')[-1].split('-f=')[0]
        distill_config = attn_mlp_checkpoint_path.split('-d=')[-1].split('-m=')[0]
    except Exception as e:
        model_config = finetune_checkpoint_path.split('-m=')[-1].split('-f=')[0]
        distill_config = None
    
    model_config = join(config_dir, 'model', f'{model_config}.yaml')
    model_config = OmegaConf.load(model_config)
    
    if distill_config is not None:
        distill_config = join(config_dir, 'experiment', f'{distill_config}.yaml')
        distill_config = OmegaConf.load(distill_config)
    else:
        distill_config = {}

    finetune_config = finetune_checkpoint_path.split('-f=')[-1].split('-')[0]
    finetune_config = join(config_dir, 'experiment', f'{finetune_config}.yaml')
    finetune_config = OmegaConf.load(finetune_config)

    # Load initial model
    model_loader = get_pretrained_loader(**model_config.model, 
                                         huggingface_token=huggingface_token)
    tokenizer = model_loader.load_tokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    model = model_loader.load(model_config['attention']['attention_type'])
    try:
        model.state_chunk_len = model_config['attention']['state_chunk_len']
    except:
        pass
    if debug:
        print_header('Pretrained Model')
        print(model)

    # Add subquadratic attentions
    model, distill_peft_config = load_and_convert_attns(model, model_config,
                                                        attention_type=None,  # in model_config
                                                        checkpoint_path=attn_mlp_checkpoint_path,
                                                        print_model=debug,
                                                        merge_loras=False,
                                                        peft_gradient_checkpointing=False,
                                                        train_attention=False)
    
    # Add PEFT parameters
    model, ft_peft_config = load_and_convert_finetune(model, finetune_config,
                                                      checkpoint_path=finetune_checkpoint_path,
                                                      print_model=debug,
                                                      merge_loras=False,
                                                      peft_gradient_checkpointing=False)
    if print_model:
        print_header('*** Model after checkpoint load ***')
        print(model)

    return model, model_config, tokenizer


def get_model_name(attn_mlp_checkpoint_path: str, finetune_checkpoint_path: str, 
                   model_config: str = None):
    model_name = '🦔 ' if attn_mlp_checkpoint_path is not None else ''
    if 'llama3_8b_' in finetune_checkpoint_path:
        model_name += f'Llama-3-8B'
    elif 'llama2_7b_' in finetune_checkpoint_path:
        model_name += f'Llama-2-7B'
    elif 'mistral_7b_' in finetune_checkpoint_path:
        model_name += f'Mistral-7B'

    if attn_mlp_checkpoint_path is not None:
        model_name += f'-Hedgehog'

    if 'alpaca_clean' in finetune_checkpoint_path:
        model_name += f'-Alpaca'

    elif model_config is not None:
        if 'llama3_8b_' in model_config:
            model_name += f'Llama-3-8B'
        elif 'llama2_7b_' in model_config:
            model_name += f'Llama-2-7B'
        elif 'mistral_7b_' in model_config:
            model_name += f'Mistral-7B'

    return model_name


def main():
    args = get_args()
    seed_everything(args.seed)
    model, model_config, tokenizer = load_model_from_checkpoint(
        args.attn_mlp_checkpoint_path, args.finetune_checkpoint_path, 
        config_dir=args.config_dir, print_model = args.print_model, debug = args.debug,
    )
    model.eval()
    model_name = get_model_name(args.attn_mlp_checkpoint_path, 
                                args.finetune_checkpoint_path, 
                                model_config)

    # Load data
    # datasets = load_mmlu(cache_dir=CACHE_DIR)
    mmlu = MMLU(num_shots=args.num_shots, cache_dir=CACHE_DIR, split=args.split)
    samples = mmlu.load_prompts()

    total = 0
    correct = 0
    for ix, (prompt, answer) in enumerate(samples):
        with torch.no_grad():
            model_input = tokenizer([prompt] * args.num_generations, 
                                    return_tensors="pt").to(model.device)
            model_output = model.generate(**model_input, use_cache=True, 
                                          max_new_tokens=args.max_new_tokens, 
                                          do_sample=True,
                                          top_k=args.top_k,
                                          top_p=args.top_p,
                                          num_return_sequences=1,
                                          pad_token_id=tokenizer.eos_token_id)

            outputs = tokenizer.batch_decode(model_output)
            print('-' * 10 + f' Sample {ix} ' + '-' * 10)
            input_seq_len = model_input['input_ids'].shape[-1]

            for _ix, _output in enumerate(model_output):
                if _ix == 0:
                    decoded_output = tokenizer.decode(_output[:input_seq_len])
                    print(decoded_output)
                    
                print('---' + f' Prediction {_ix} ' + '---')
                decoded_output = tokenizer.decode(_output[input_seq_len:])
                print(decoded_output)
                print('')
            print('---' + f' True Answer {ix} ' + '---')
            print(f' {answer}')

            # Compute greedy answer
            model_input = tokenizer([prompt], return_tensors="pt").to(model.device)
            model_output = model.generate(**model_input, use_cache=True, 
                                          max_new_tokens=args.max_new_tokens, 
                                          do_sample=False,
                                          num_return_sequences=1,
                                          pad_token_id=tokenizer.eos_token_id)
            outputs = tokenizer.batch_decode(model_output)
            input_seq_len = model_input['input_ids'].shape[-1]
            for _ix, _output in enumerate(model_output):
                print('---' + f' Greedy Pred {ix} ' + '---')
                decoded_output = tokenizer.decode(_output[input_seq_len:])
                print(decoded_output)
                print('')
                if decoded_output.replace(' ', '').upper() == answer.replace(' ', '').upper():
                    correct += 1
            total += 1
    print(f'Final MMLU acc: {correct / total * 100:.4f}% ({correct} / {total})')

if __name__ == '__main__':
    main()
