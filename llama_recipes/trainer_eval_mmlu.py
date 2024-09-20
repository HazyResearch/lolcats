# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
"""
Training and evaluation functions for evaluating MMLU following LM Evaluation Harness Implementation
- Modified from https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/utils/train_utils.py

"""
import os
import time
from contextlib import nullcontext
from datetime import datetime

import torch
# import torch.cuda.nccl as nccl
import torch.distributed as dist
# from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm

# Ours
import numpy as np
from accelerate.utils import is_xpu_available
from llama_recipes.utils.memory_utils import MemoryTrace
from llama_recipes.trainer_attention import (
    eval_loop, clear_gpu_cache, save_to_json,
    setup, setup_environ_flags,  # imports into distill_llama_finetune.py
    print_model_size, get_policies
)


class MMLUComputer():
    """
    Computes the loss for next-token prediction
    """
    def __init__(self, **kwargs: any):
        super().__init__()
        # self.categories = {}  # {'mmlu-category': {'correct': int, 'total': int, 'acc': float}}
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def compute_mmlu_loss(self, model: torch.nn.Module, data: torch.Tensor, 
                          rank: int = 0, local_rank: int = 0):
        """Compute MMLU loss"""
        
        input_keys = {'input_ids'}  # , 'attention_mask'} (assume packing / no padding)
        inputs = {k: v.to(model.device) for k, v in data.items() if k in input_keys}  
        outputs = model(**inputs, output_attentions=False, use_cache=False)  # use_cache=False)
        
        outputs = outputs.get('logits')[..., -2, :].contiguous()  # b, d
        targets = data.get('input_ids')[..., -1].contiguous().to(outputs.device)  # b, d
        # Compute cross-entropy loss
        losses = []
        for choice_idx in range(outputs.shape[0]):
            losses.append(self.criterion(outputs[choice_idx], targets[choice_idx]))
        losses = torch.stack(losses).cpu()  # b, 1
        pred = torch.argmin(losses, dim=0)
        correct = data['target'].cpu() == pred
        return losses, correct, data['category']


def evaluate_mmlu(model, train_config, eval_dataloader,
                  local_rank, tokenizer, wandb_run, 
                  epoch: int = None, rank: int = 0):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_epoch_loss
    """
    loss_computer = MMLUComputer(**train_config.trainer)
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()

    metrics = {
        'all': {'total': 0, 'correct': 0}
        # categories: {'total': 0, 'correct': 0}
    }
    
    _epoch = f' {epoch}' if epoch is not None else ''
    pbar = tqdm(eval_dataloader, colour="green", desc=f"Rank {rank} | Eval Epoch{_epoch}", dynamic_ncols=True)
    for step, batch in enumerate(pbar):
        for key in batch.keys():
            if train_config.enable_fsdp:
                batch[key] = batch[key].to(local_rank)
            else:
                if is_xpu_available():
                    batch[key] = batch[key].to('xpu:0')
                else:
                    batch[key] = batch[key].to('cuda:0')
                    
        # Ensure no gradients are computed for this scope to save memory
        with torch.no_grad():
            # Forward pass and compute loss
            losses, _correct, _category = loss_computer.compute_loss(model, batch, rank=rank, 
                                                                     local_rank=local_rank)
            metrics['all']['total'] += 1
            metrics['all']['correct'] += _correct
            
            if _category in metrics:
                metrics[_category]['total'] += 1
                metrics[_category]['total'] += _correct
            else:
                metrics[_category]['total'] = 1
                metrics[_category]['total'] = _correct

            total = metrics['all']['total']
            correct = metrics['all']['correct']
            total_acc = total / correct * 100
                
        pbar.set_description(f"Rank {rank} | Eval Epoch{_epoch} | total acc: {total_acc:.3f}% ({total} / {correct})")

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
        for category in metrics:
            for k, v in metrics[category]:
                dist.all_reduce(v, op=dist.ReduceOp.SUM)
    elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:  # what's the diff b/t this condition and above?
        for category in metrics:
            for k, v in metrics[category]:
                dist.all_reduce(v, op=dist.ReduceOp.SUM)

    for k in metrics:
        metrics[k]['acc'] = metrics[k]['correct'] / metrics[k]['total']

    eval_epoch_ppl = torch.exp(eval_epoch_loss).item()
    eval_epoch_loss = eval_epoch_loss.item()
    del eval_loss; del batch
    clear_gpu_cache()

    # Print evaluation metrics
    if local_rank == 0 or not train_config.enable_fsdp:
        print(f" eval_epoch_loss={eval_epoch_loss}, eval_epoch_ppl={eval_epoch_ppl}")

    if wandb_run:
        wandb_run.log(metrics)

    return metrics
