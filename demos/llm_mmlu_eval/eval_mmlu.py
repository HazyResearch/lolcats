
import os
from os.path import join
import dataclasses
import random
from tqdm import tqdm
import pickle

from accelerate.utils import is_xpu_available
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from omegaconf import OmegaConf

import sys 
sys.path.append('./llama_recipes/')
sys.path.append('./src')
sys.path.append('./../src')

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.utils.fsdp_utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import update_config
from llama_recipes.trainer_finetune import (
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)
from llama_recipes.model_checkpointing.distill_checkpoint_handler import load_sharded_model_single_gpu
from llama_recipes.distill_llama import setup_wandb, setup_fsdp_config

from src.utils.setup import update_config_from_args, update_model_config_from_args
from src.utils.logging import print_header, print_config
from src.finetune import prepare_finetune_configs 
from src.model.pretrained import get_pretrained_loader
from src.model.load_model import load_and_convert_attns, load_and_convert_finetune
import argparse

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Please pip install huggingface-hub")


def get_args():
    """Get attention transfer args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default='lolcats')
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--distill_config", type=str, default=None)
    parser.add_argument("--finetune_config", type=str, default=None)
    parser.add_argument("--eval_config", type=str, default=None)

    parser.add_argument("--layers_per_model", type=int, default=None)
    parser.add_argument("--layers_limit", type=int, default=None)
    parser.add_argument("--layers_min_limit", type=int, default=None)

    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--load_distill_checkpoint", type=str, default=None)
    parser.add_argument("--load_finetune_checkpoint", type=str, default=None)
    parser.add_argument("--finetune_checkpoint_path", type=str, default=None)
    parser.add_argument("--resume_distill", action='store_true', default=None)
    parser.add_argument("--resume_finetune", action='store_true', default=None)

    # Override default configs
    # Feature map / model
    parser.add_argument("--attention_type", type=str, default=None)
    parser.add_argument("--learned_kernel", type=str, default=None)
    parser.add_argument("--lk_skip_connection", action='store_true', default=None)
    parser.add_argument("--lk_zero_init", action='store_true', default=None)
    parser.add_argument("--tie_qk_kernels", action='store_true', default=None)
    parser.add_argument("--train_qk", action='store_true', default=None)
    parser.add_argument("--state_chunk_len", type=int, default=None)
    
    # Training
    ## Distributed training / Llama recipes
    parser.add_argument("--enable_fsdp", action='store_true', default=None)
    parser.add_argument("--low_cpu_fsdp", action='store_true', default=None)
    parser.add_argument("--pure_bf16", action='store_true', default=None)
    parser.add_argument("--fsdp_activation_checkpointing", action='store_true', default=None)
    parser.add_argument("--fsdp_cpu_offload", action='store_true', default=None)
    
    ## Hyperparameters
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--optim", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--no_peft_grad_ckpt", action='store_true', default=None)

    # Finetuning
    parser.add_argument("--finetune_lr", type=float, default=None)
    parser.add_argument("--finetune_attn_mlps", action='store_true', default=None)
    
    # Dataloading
    parser.add_argument("--dataset_chunk_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)

    # Evaluation
    parser.add_argument("--no_init_eval", action='store_true', default=False)
    parser.add_argument("--eval_steps", type=int, default=None)

    # Experimental tag for saving mmlu preds
    parser.add_argument("--experiment_tag", type=str, default=None)

    # Miscellaneous
    parser.add_argument("--huggingface_token", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default='./checkpoints')
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action='store_true', default=None)
    parser.add_argument("--no_cuda", action='store_true', default=None)
    parser.add_argument("--no_wandb", action='store_true', default=None)
    parser.add_argument("--wandb_entity", type=str, default='hazy-research')
    parser.add_argument("--debug", action='store_true', default=None)
    parser.add_argument("--num_train_steps", type=int, default=-1)

    # DEMO
    ## Generation
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    ## Miscellaneous
    parser.add_argument("--benchmark", action='store_true', default=False)
    parser.add_argument("--print_model", action='store_true', default=False)

    args = parser.parse_args()

    distill_name = args.distill_config
    finetune_name = args.finetune_config

    args.run_name = f'dl-d={distill_name}-m={args.model_config}-f={finetune_name}'
    if args.no_peft_grad_ckpt is not None:
        args.run_name += f'-npgc={args.no_peft_grad_ckpt}'
    if args.fsdp_activation_checkpointing is not None:
        args.run_name += f'-fac={args.fsdp_activation_checkpointing}'

    if args.debug:
        args.run_name += '-debug'
    
    args.run_name = args.run_name.replace('True', '1').replace('False', '0')  # concise hacks
    return args


class InputDataset(Dataset):
    def __init__(self, data):
        self.samples = data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        entry = x[-1]
        input_tokens = torch.tensor(entry['token_ids'])
        ans_tokens = torch.tensor(entry['answer_token_ids'])
        text = entry['text']
        text, answer_choice = text

        if 'A' in answer_choice:
            answer_choice = 0
        elif 'B' in answer_choice:
            answer_choice = 1
        elif 'C' in answer_choice:
            answer_choice = 2
        elif 'D' in answer_choice:
            answer_choice = 3
        else:
            raise ValueError(f"Invalid answer choice: {answer_choice}")
        
        gold_answer = x[2]['gold']
        task_name = x[1]
        query = x[2]['query']
        doc_id = f"{task_name}-{query}"
        
        return {
            'input_ids': input_tokens, 
            'target_ids': ans_tokens, 
            'doc_id': doc_id, 
            'answer_choice': x[0],
            'answer': gold_answer
        }


def evaluate_lm(model, train_config, eval_dataloader,
                local_rank, rank: int = 0, TAG=None):

    if TAG is None:
        print(f"TAG is None, Will not save the predictions out.")

    if rank == 0:
        for n, p in model.named_parameters():
            if ('layers.0.' in n and 'base_attn' not in n and 
                '.0.mlp.' not in n and '.block_sparse_moe' not in n):
                print(f'-> {n}:\n', p)

    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()

    log_probabilities = defaultdict(dict)
    answers = {}
    pbar = tqdm(eval_dataloader,colour="green", desc=f"Rank {rank}", dynamic_ncols=True)
    correctness = {}
    skipped = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    for step, batch in enumerate(pbar):
        for key in batch.keys():
            if (type(batch[key]) == torch.Tensor):
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    if is_xpu_available():
                        batch[key] = batch[key].to('xpu:0')
                    else:
                        batch[key] = batch[key].to('cuda:0')
        
        # Ensure no gradients are computed for this scope to save memory
        with torch.no_grad():
            input_keys = {'input_ids'}  
            inputs = {k: v.to(model.device) for k, v in batch.items() if k in input_keys} 
            bs = inputs['input_ids'].shape[0] 
            seq_len = inputs['input_ids'].shape[1]

            # model call
            outputs = model(**inputs, output_attentions=False, use_cache=False) 
            outputs = outputs.get('logits')[..., -1, :].contiguous()
            target_ids = batch['target_ids'].to(model.device)

            # find the answer with the highest probability
            losses = []
            for choice_idx in range(outputs.shape[0]):
                output = outputs[choice_idx].unsqueeze(0)
                target = target_ids[choice_idx].view(-1)
                losses.append(criterion(output, target))
            losses = torch.stack(losses).cpu()  # b, 1
            pred = torch.argmin(losses, dim=0)
            answer = batch['answer']
            if type(pred) == torch.Tensor:  # Flagging this logic.
                pred = pred.item()
            if rank == 0:
                print(f"--> {step=}: {answer=}, {pred=}")
            correct = (answer[0].cpu() == pred)
            correctness[step] = correct
            skipped[step] = False

        # free up memory
        del outputs; del inputs; del target_ids; del losses; del pred; del correct

        if step % 100 == 0 and rank == 0:
            total_correct = sum(correctness.values())
            print(f"--> at step {step}, Total Correct: {total_correct}/{len(correctness)}")
            if TAG is not None:
                with open(f"{TAG}_{step}_mmlu_predictions_{rank}.pkl", 'wb') as f:
                    pickle.dump(correctness, f)
                with open(f"{TAG}_{step}_mmlu_skipped_{rank}.pkl", 'wb') as f:
                    pickle.dump(skipped, f)

    if rank == 0:
        total_correct = sum(correctness.values())
        print(f"--> at step {step}, Total Correct: {total_correct}/{len(correctness)}")
        if TAG is not None:
            with open(f"{TAG}_{step}_mmlu_predictions_{rank}.pkl", 'wb') as f:
                pickle.dump(correctness, f) 
            with open(f"{TAG}_{step}_mmlu_skipped_{rank}.pkl", 'wb') as f:
                pickle.dump(skipped, f)

    del log_probabilities; del batch
    clear_gpu_cache()

    total_correct = sum(correctness.values())
    mmlu_score = total_correct / len(correctness)
    return mmlu_score


def load_hf_weights(model, distill_repo_id, ft_repo_id, filename="model.pt"):
    for repo_id in [distill_repo_id, ft_repo_id]:
        if repo_id is None: continue 
        print(f"Loading weights from {repo_id}")
        local_file_path = hf_hub_download(repo_id=repo_id, filename=filename)    
        state_dict = torch.load(local_file_path)
        if 'model_state_dict' in state_dict: 
            state_dict = state_dict['model_state_dict']
        else:
            pass
        _keys = model.load_state_dict(state_dict, strict=False)
        if len(_keys.unexpected_keys) > 0:
            new_state_dict = {k.replace('model.', 'model.model.'): v for k, v in state_dict.items()}
            _keys = model.load_state_dict(new_state_dict, strict=False)
        if len(_keys.unexpected_keys) > 0:
            new_state_dict = {k.replace('model.', 'base_model.model.model.'): v for k, v in state_dict.items()}
            _keys = model.load_state_dict(new_state_dict, strict=False)
        try:
            assert len(_keys.unexpected_keys) == 0
            print_header('*** All expected keys matched successfully ***')
        except Exception as e:
            print(e)
            print_header('*** Error: unexpected keys in checkpoint - please fix ***')
            print('Unexpected keys:')
            for k in _keys.unexpected_keys:
                print(k)
            exit()
    return model


def main():
    # ---------
    # 1. SET UP
    # ---------
    args = get_args()
    args.checkpoint_dir = join(args.checkpoint_dir, args.model_config)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    kwargs = vars(args)

    # Load distillation + attention configs
    distill_config_path = join('./configs/experiment', f'{args.distill_config}.yaml')
    distill_config = OmegaConf.load(distill_config_path)
    distill_config = update_config_from_args(distill_config, args)

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

    # Update dataset pretrained model config
    for k in distill_config.dataset.pretrained_model_config:
        print(f"{k=}")
        distill_config.dataset.pretrained_model_config[k] = getattr(model_config.model, k)

    args.run_name = args.run_name.replace('True', '1').replace('False', '0')  # concise hacks
    
    # Update the configuration for the training and sharding process
    distill_config = setup_fsdp_config(distill_config, args, 'distill')  # patch llama-recipes args
    fsdp_config = FSDP_CONFIG()
    update_config((fsdp_config), **vars(args))
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"{distill_config.dataset.pretrained_model_config=}")
        print("*****"*10)
        print(f"{model_config.model=}")

    if rank == 0 or not args.enable_fsdp:
        print_header('Distillation Config')
        print_config(distill_config)
        print_header('Model Config')
        print_config(model_config)
        print_header('FSDP Config')
        print_config(dataclasses.asdict(fsdp_config))

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
            wandb_run = setup_wandb(distill_config, fsdp_config, **kwargs)  


    finetune_checkpoint_path = args.finetune_checkpoint_path
    if finetune_checkpoint_path == "None":
        finetune_checkpoint_path = None

    # ------------------------
    # 2. LOAD PRETRAINED MODEL
    # ------------------------
    # Load the pre-trained model and setup its configuration
    # Initialize tokenizer and model loader
    model_loader = get_pretrained_loader(**model_config.model)
    tokenizer = model_loader.load_tokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    model_type = "softmax"
    if 'lama' in model_config.model.pretrained_model_name_or_path:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer as DecoderLayer
        from src.model.modeling_llama import LolcatsLlamaForCausalLM as ModelClass
        if finetune_checkpoint_path is None: 
            model_type = 'softmax' 
        else:
            model_type = 'llama'
    if rank == 0:
        print(f"{model_type=}")

    # Convert model
    if finetune_checkpoint_path is not None: 
        try:
            args.attention_type = model_config['attention']['attention_type']
        except AttributeError:
            args.attention_type = 'lolcats_llama'
    else:
        args.attention_type = "softmax"
    
    model = model_loader.load(args.attention_type)
    model.state_chunk_len = model_config['attention']['state_chunk_len']
    model_config.model_name = model_config.model.pretrained_model_name_or_path
    print_model_size(model, model_config, rank if args.enable_fsdp else 0)
    if args.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    # -------------------------------
    # 3. CONVERT DISTILLED ATTENTIONS
    # -------------------------------
    print(f"Before convert attns")
    if finetune_checkpoint_path is not None:
        model, distill_peft_config = load_and_convert_attns(model, model_config,
                                                        attention_type=args.attention_type,
                                                        checkpoint_path=None, 
                                                        print_model=args.verbose,
                                                        merge_loras=False,
                                                        peft_gradient_checkpointing=not args.no_peft_grad_ckpt,
                                                        train_attention=False,
                                                        rank=rank)
    else: 
        distill_peft_config = None 

    if rank == 0:
        print(model)
    if rank == 0:
        print_header('** Sanity check model weights **')
        for n, p in model.named_parameters():
            if ('layers.0.' in n and ('feature_map' in n or 'lora' in n)):
                print(f'-> {n}:\n', p)
    
    if wandb_run and distill_peft_config is not None:
        wandb_run.config.update(distill_peft_config)
        
    # ----------------------------
    # 4. ADD FINETUNING PARAMETERS
    # ----------------------------
    finetune_config, args = prepare_finetune_configs(args, model_config,  args.finetune_config)
    finetune_config = setup_fsdp_config(finetune_config, args, 'finetune')
    if args.finetune_lr is not None:
        finetune_config.model_name += f'=flr={args.finetune_lr}'
            
    if rank == 0:
        print(f"{args.load_finetune_checkpoint=}")
    if finetune_checkpoint_path is not None:
        model, _ = load_and_convert_finetune(model, finetune_config,
                                            #  checkpoint_path=args.load_finetune_checkpoint,
                                            print_model=args.verbose,
                                            merge_loras=False,
                                            peft_gradient_checkpointing=not args.no_peft_grad_ckpt,
                                            rank=rank)

    # Load in our trained weights, assumes path has weights from both attention transfer and LoRA
    if finetune_checkpoint_path is not None:
        if '.pt' in args.finetune_checkpoint_path:
            with torch.no_grad():
                _keys = model.load_state_dict(torch.load(args.finetune_checkpoint_path), strict=False)
                if rank == 0:
                    print(f"Found {len(_keys.unexpected_keys)} unexpected keys.")
        elif 'hazyresearch' in args.finetune_checkpoint_path:
            print(f"Loading from huggingface.")
            model = load_hf_weights(model, args.finetune_checkpoint_path, None, filename="model.pt")
        else:
            model = load_sharded_model_single_gpu(model, model_path=args.finetune_checkpoint_path,  
                                                    cfg=finetune_config, rank=rank)

    if rank == 0:
        print(f"{args.enable_fsdp=}")
    if rank == 0 or not args.enable_fsdp:  # debugging
        print_header('** Sanity check model weights **')
        for n, p in model.named_parameters():
                if ('layers.0.' in n and 'base_attn' not in n and 
                    '.0.mlp.' not in n and '.block_sparse_moe' not in n):
                    print(f'-> {n}:\n', p)

    # ------------------------------------------------------
    # 5. SETUP FSDP AND LOAD DISTILLED ATTENTION CHECKPOINTS
    # ------------------------------------------------------
    if args.enable_fsdp:

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank, model=model_type)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, DecoderLayer)
        
        device_id = 0
        if is_xpu_available():
            device_id = torch.xpu.current_device()
        elif torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            print('-> device_id:', device_id)
            print(f"Model")
            if rank == 0:
                print(model)

        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrapping_policy, 
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=args.low_cpu_fsdp,  
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if args.low_cpu_fsdp and rank != 0 else None,
        )

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

    if rank == 0:
        print(f"Getting the MMLU dataset:")
    if not os.path.exists("demos/llm_mmlu_eval/mmlu.pkl"):
        print(f"Please make sure the paths are set corretly for mmlu.pkl")
    with open("demos/llm_mmlu_eval/mmlu.pkl", 'rb') as f:
        data = pickle.load(f)
    dataset = InputDataset(data)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=4)
    if not args.enable_fsdp or rank == 0:
        print(f"--> Validation Set Length = {len(dataloader.dataset)}")
    
    if rank == 0:
        print(f"Running evaluation:")

    TAG = args.experiment_tag
    mmlu_score = evaluate_lm(model, finetune_config, 
        dataloader,
        local_rank if args.enable_fsdp else None, 
        rank = rank if args.enable_fsdp else None,
        TAG=TAG
    )
    if rank == 0:
        print(f"--> Final MMLU Score: {mmlu_score}")

if __name__ == "__main__":
    main()

