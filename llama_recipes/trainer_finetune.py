# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
"""
Training and evaluation functions for finetuning after attention transfer
- Modified from https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/utils/train_utils.py

We do 
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


class LossComputer():
    """
    Computes the loss for next-token prediction
    """
    def __init__(self, **kwargs: any):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def compute_loss(self, model: torch.nn.Module, data: torch.Tensor, 
                     rank: int = 0, local_rank: int = 0):
        """Compute the loss for attention distillation"""
        input_keys = {'input_ids'}  # , 'attention_mask'} (assume packing / no padding)
        inputs = {k: v.to(model.device) for k, v in data.items() if k in input_keys}  
        outputs = model(**inputs, output_attentions=False, use_cache=False)  # use_cache=False)
        outputs = outputs.get('logits')[..., :-1, :].contiguous()
        targets = data.get('labels')[..., 1:].contiguous()
        # Flatten and compute cross-entropy loss
        outputs = outputs.view(-1, outputs.shape[-1])
        targets = targets.view(-1).to(outputs.device)
        if (targets != -100).sum() == 0:
            return torch.Tensor([0])[0]
        else:
            loss = self.criterion(outputs, targets)
            targets = targets.cpu()
            outputs = outputs.cpu()
            # print(f'rank: {rank} | local_rank: {local_rank} | loss: {loss.item():.5f} | shape: {targets.shape} |')
            return loss  # , {'ppl': np.exp(loss.item()), 'seq_len': targets.shape[-1] + 1}


def train(model, train_dataloader, eval_dataloader, tokenizer,
          optimizer, lr_scheduler, gradient_accumulation_steps,
          train_config, fsdp_config=None, local_rank=None, rank=None,
          wandb_run=None, stepwise_scheduler=False) -> dict[torch.Tensor]:
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: 
        results dictionary containing average training and validation loss
        best_checkpoint_path: The path to the best checkpoint
    """
    loss_computer = LossComputer(**train_config.trainer)

    if rank == 0 or rank is None:
        print('-> Gradient accumulation steps:', gradient_accumulation_steps)
        print('-> Total # of training samples:', len(train_dataloader))
        total_length = len(train_dataloader)//gradient_accumulation_steps
        print('-> Total # of training updates:', total_length)

    # print('-> loss_computer:', loss_computer)
    
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"]) 
        # print('-> world_size:', world_size)

    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_loss = []
    val_loss =[]

    if train_config.save_metrics:
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []
        # print(f'-> Saving metrics to {metrics_filename}')
        
    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    best_checkpoint_path = None
    total_step = 0

    for epoch in range(train_config.num_epochs):
        epoch_start_time = time.perf_counter()
        # print('-> epoch:', epoch)
        # if True:
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            print(f'-> Model is training on rank {rank}')
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            for step, batch in enumerate(train_dataloader):
                model.train()
                # print('-> step:', step)
                for key in batch.keys():
                    if train_config.enable_fsdp:
                        if is_xpu_available():
                            batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                        else:
                            batch[key] = batch[key].to(local_rank)
                    else:
                        if is_xpu_available():
                            batch[key] = batch[key].to('xpu:0')
                        else:
                            batch[key] = batch[key].to('cuda:0')
                with autocast():
                    loss = loss_computer.compute_loss(model, batch, rank, local_rank)

                train_step_loss.append(loss.item())
                train_step_perplexity.append(float(np.exp(loss.item())))
                loss = loss / gradient_accumulation_steps
                total_loss += loss.detach().float()

                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    if (total_step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                            scaler.unscale_(optimizer)
                            if train_config.enable_fsdp:
                                model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        pbar.update(1)
                        if stepwise_scheduler:
                            lr_scheduler.step()
                else:
                    # regular backpropagation when fp16 is not used
                    # if loss.sum() > 0:  # hack for non-answer targets
                    loss.backward()
                    if (total_step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                            if train_config.enable_fsdp:
                                model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                        optimizer.step()
                        optimizer.zero_grad()
                        pbar.update(1)
                        if stepwise_scheduler:
                            lr_scheduler.step()

                if wandb_run: 
                    if not train_config.enable_fsdp or rank==0:
                        wandb_run.log({
                            'train/epoch': epoch + 1,
                            'train/step': total_step,  # epoch * len(train_dataloader) + step,
                            'train/loss': train_step_loss[-1],
                            'train/ppl': train_step_perplexity[-1],
                            'train/lr': optimizer.param_groups[-1]['lr']
                        })

                metrics = f"loss: {train_step_loss[-1]:.5f} | lr: {optimizer.param_groups[0]['lr']:.5f} | ppl: {train_step_perplexity[-1]}"
                # pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float():.5f}, lr: {optimizer.param_groups[0]['lr']:.5f})")
                pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed ({metrics})")

                if train_config.save_metrics:
                    save_to_json(metrics_filename, train_step_loss, train_loss, val_step_loss, val_loss)

                if total_step == getattr(train_config, 'num_train_steps', -1):
                    break  # Early exit for debugging later logic

                if (train_config.run_validation and (
                    (total_step + 1) % (train_config.eval_steps * gradient_accumulation_steps) == 0)):  #  or step == len(train_dataloader) - 1)):
                    dist.barrier()
                    eval_outputs = eval_loop(model, evaluate_lm, optimizer, lr_scheduler,
                                             train_config, fsdp_config, rank, eval_dataloader,
                                             local_rank, tokenizer, wandb_run,
                                             val_step_loss, val_loss, best_val_loss,
                                             checkpoint_times, epoch, total_step)
                    dist.barrier()
                    save_path, val_step_loss, val_loss, best_val_loss, checkpoint_times = eval_outputs
                    if save_path is not None:
                        best_checkpoint_path = save_path
                    model.train()
                    print(f'-> Model is training on rank {rank}')
                total_step += 1
            pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        
        # Reducing total_loss across all devices if there's more than one CUDA device
        if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader) * gradient_accumulation_steps
        train_epoch_loss = total_loss / len(train_dataloader) * gradient_accumulation_steps
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)
        train_loss.append(float(train_epoch_loss))
        
        if not train_config.enable_fsdp or rank==0:
            memtrace.print_stats()

        # Update the learning rate as needed
        # lr_scheduler.step()
        dist.barrier()
        eval_outputs = eval_loop(model, evaluate_lm, optimizer, lr_scheduler,
                                 train_config, fsdp_config, rank, eval_dataloader,
                                 local_rank, tokenizer, wandb_run,
                                 val_step_loss, val_loss, best_val_loss,
                                 checkpoint_times, epoch, total_step)
        dist.barrier()
        save_path, val_step_loss, val_loss, best_val_loss, checkpoint_times = eval_outputs
        if save_path is not None:
            best_checkpoint_path = save_path

        if rank == 0 or not train_config.enable_fsdp:
            print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

    results = {'best_val_loss': best_val_loss, 
               'checkpoint_times': sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0}
    return results, best_checkpoint_path


def evaluate_lm(model, train_config, eval_dataloader,
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
    loss_computer = LossComputer(**train_config.trainer)
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    val_step_loss = []

    eval_loss = 0.0  # Initialize evaluation loss
    _epoch = f' {epoch}' if epoch is not None else ''
    pbar = tqdm(eval_dataloader,colour="green", desc=f"Rank {rank} | Eval Epoch{_epoch}", dynamic_ncols=True)
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
            loss = loss_computer.compute_loss(model, batch, rank=rank, local_rank=local_rank)
            if train_config.save_metrics:
                val_step_loss.append(loss.detach().cpu().float().item())

            # Check NaNs in loss
            if torch.isnan(loss).any():
                print("NaN detected in eval loss. Skipping evaluation accumulation.")
            else:
                eval_loss += loss.detach().float()
            _ppl = torch.exp(eval_loss/(step+1)).item()
        pbar.set_description(f"Rank {rank} | Eval Epoch{_epoch} | step_loss: {loss.item():.5f} | avg_loss: {eval_loss.item()/(step+1):.5f} | avg_ppl: {_ppl:.5f}")
        if step > 20:  # hack
            break

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:  # what's the diff b/t this condition and above?
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss
    # print('len(eval_dataloader):', len(eval_dataloader))
    # print('step + 1:', step + 1)
    # print('world_size:', world_size)
    eval_epoch_loss = eval_loss / 20  # len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size

    eval_epoch_ppl = torch.exp(eval_epoch_loss).item()
    eval_epoch_loss = eval_epoch_loss.item()
    del eval_loss; del batch
    clear_gpu_cache()

    # Print evaluation metrics
    if local_rank == 0 or not train_config.enable_fsdp:
        print(f" eval_epoch_loss={eval_epoch_loss}, eval_epoch_ppl={eval_epoch_ppl}")

    if wandb_run:
        wandb_run.log({'eval/loss': eval_epoch_loss, 'eval/ppl': eval_epoch_ppl}, commit=False)

    return eval_epoch_loss, val_step_loss
