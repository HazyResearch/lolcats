# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
"""
Training and evaluation functions for attention distillation
- Modified from https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/utils/train_utils.py
"""

import os
import time
import json
from datetime import timedelta
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime
from pkg_resources import packaging
import yaml

import torch
import torch.nn as nn
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer

# Ours
from llama_recipes.model_checkpointing.distill_checkpoint_handler import (
    save_model_checkpoint, 
    save_model_and_optimizer_sharded, 
    save_optimizer_checkpoint
)
from llama_recipes.policies import fpSixteen,bfSixteen # get_llama_wrapper
from llama_recipes.policies import get_llama_wrapper, get_mistral_wrapper, get_mixtral_wrapper
from llama_recipes.utils.memory_utils import MemoryTrace
from accelerate.utils import is_xpu_available, is_ccl_available


def set_tokenizer_params(tokenizer: LlamaTokenizer):
    """Set the tokenizer parameters for padding"""
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"


def byte2mb(x: float) -> int:
    """Converts bytes to megabytes"""
    return int(x / 2**20)


class LossComputer():
    """
    Computes the loss for attention distillation
    """
    def __init__(self, mse_factor: int = 1000, xent_factor: int = 1, 
                 n_queries: int = None, n_keys: int = None, **kwargs: any) -> None:
        super().__init__()
        self.criterion_mse = nn.MSELoss(reduction='mean')
        self.criterion_xent = nn.CrossEntropyLoss(reduction='mean')
        self.mse_factor = mse_factor
        self.xent_factor = xent_factor
        self.n_queries = n_queries
        self.n_keys = n_keys


    def compute_loss(self, model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Compute the loss for attention distillation"""
        loss = 0
        loss_mse = 0
        loss_xent = 0
        n_layers = 0  # Number of layers to distill
        outputs = model(**inputs, output_attentions=True, use_cache=False).get('attentions')        
        for _, attns in enumerate(outputs):
            # ((a_pred, a_true), (y_pred, _y_true))

            if attns is not None:

                # attention_pred, attention_true (probability distributions)
                if self.xent_factor > 0:
                    # Cross-entropy loss
                    a_pred, a_true = attns[0]
                    if self.n_queries is not None:
                        a_pred = a_pred[:, :, -self.n_queries:, :]
                        a_true = a_true[:, :, -self.n_queries:, :]
                    a_pred = a_pred.clamp(min=1e-12).log()  # nn.CrossEntropy assumes unnormalized logits
                    k_len = a_true.shape[-1]  # batch, n_heads, q_len, k_len
                    # Compute mean cross-entropy over all queries
                    a_pred = a_pred.contiguous().view(-1, k_len)
                    a_true = a_true.contiguous().view(-1, k_len)
                    loss_xent += self.criterion_xent(a_pred[:, :self.n_keys],
                                                     a_true[:, :self.n_keys])
                    # a_pred = a_pred.detach().cpu()
                    # a_true = a_true.detach().cpu()
                    # loss_xent += self.criterion_xent(a_pred.to(model.device), 
                    #                                  a_true.to(model.device))

                # y_preds, y_true (raw values)
                if self.mse_factor > 0:
                    # 
                    loss_mse += self.criterion_mse(*attns[1]) 
                    # attns[1][0] = attns[1][0].detach().cpu()
                    # attns[1][1] = attns[1][1].detach().cpu()
                    # loss_mse += self.criterion_mse(*[a.to(model.device) for a in attns[1]])
                n_layers += 1
                # torch.cuda.empty_cache()
        
        if n_layers > 0:
            loss_xent = loss_xent * self.xent_factor / n_layers
            loss_mse = loss_mse * self.mse_factor / n_layers
        
        if ( type(loss_xent) == float ): 
            loss = loss_mse
        elif ( type(loss_mse) == float ):
            loss = loss_xent
        else:
            loss = (loss_xent + loss_mse)
            
        try:
            loss_xent = loss_xent.item()
        except:
            pass 
        try:
            loss_mse = loss_mse.item()
        except:
            pass
        loss_metrics = {
            'loss_xent': loss_xent, 
            'loss_mse': loss_mse, 
            'loss': loss.item(),
            'xent_factor': self.xent_factor,
            'mse_factor': self.mse_factor,
        }
        return loss, loss_metrics


def train(model, train_dataloader, eval_dataloader, tokenizer,
          optimizer, lr_scheduler, gradient_accumulation_steps,
          train_config, fsdp_config=None, local_rank=None, rank=None, 
          wandb_run=None, eval_mode=False) -> tuple[dict[torch.Tensor], str]:
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
        _dt = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{_dt}.json".replace('//', '/')
        train_step_loss = []
        val_step_loss = []
        # print(f'-> Saving metrics to {metrics_filename}')

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    best_checkpoint_path = None
    for epoch in range(train_config.num_epochs):
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            if eval_mode:
                model.eval()
                print(f'-> Model is eval mode on rank {rank}')
            else:
                model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            for step, batch in enumerate(train_dataloader):
                model.train()
                # print('-> step:', step)
                for key in batch.keys():
                    if key == 'labels':
                        batch[key] = None  # don't use labels for attention distillation
                    else:
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
                    loss, loss_metrics = loss_computer.compute_loss(model, batch)
                loss = loss / gradient_accumulation_steps
                if train_config.save_metrics:
                    train_step_loss.append(loss.detach().cpu().float().item())

                total_loss += loss.detach().float()
                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    if ((step + 1) % gradient_accumulation_steps == 0 
                        or step == len(train_dataloader) - 1):
                        if (train_config.gradient_clipping 
                            and train_config.gradient_clipping_threshold > 0.0):
                            scaler.unscale_(optimizer)
                            if train_config.enable_fsdp:
                                model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                            else:
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(), train_config.gradient_clipping_threshold)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        pbar.update(1)
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if ((step + 1) % gradient_accumulation_steps == 0 
                        or step == len(train_dataloader) - 1):
                        if (train_config.gradient_clipping 
                            and train_config.gradient_clipping_threshold > 0.0):
                            if train_config.enable_fsdp:
                                model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                            else:
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(), train_config.gradient_clipping_threshold)
                        optimizer.step()
                        optimizer.zero_grad()
                        pbar.update(1)
                if wandb_run:
                    if not train_config.enable_fsdp or rank==0:
                        wandb_run.log({
                            'train/epoch': epoch + 1,
                            'train/step': epoch * len(train_dataloader) + step,
                            'train/loss': loss.detach().cpu().item(),
                        })

                desc = f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.item():.5f}, lr: {optimizer.param_groups[0]['lr']:.5f})"
                for k, v in loss_metrics.items():
                    desc += f" | {k}: {v:.5f}"
                pbar.set_description(desc)

                if train_config.save_metrics:
                    save_to_json(metrics_filename, train_step_loss, train_loss, val_step_loss, val_loss)

                if step == getattr(train_config, 'num_train_steps', -1):
                    break  # Early exit for debugging later logic

                if (train_config.run_validation and (
                    (step + 1) % (train_config.eval_steps * gradient_accumulation_steps) == 0)):  #  or step == len(train_dataloader) - 1)):
                    eval_outputs = eval_loop(model, evaluate_attn, optimizer, lr_scheduler, train_config, fsdp_config, rank,
                                             eval_dataloader, local_rank, tokenizer, wandb_run,
                                             val_step_loss, val_loss, best_val_loss, checkpoint_times, epoch, step)
                    save_path, val_step_loss, val_loss, best_val_loss, checkpoint_times = eval_outputs
                    if save_path is not None:
                        best_checkpoint_path = save_path
                    if not eval_mode:
                        model.train()
                        print(f'-> Model is training on rank {rank}')
                    del loss; del batch; del eval_outputs
                    clear_gpu_cache()
            pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_loss.append(float(train_epoch_loss))

        if not train_config.enable_fsdp or rank==0:
            memtrace.print_stats()

        # Update the learning rate as needed
        # lr_scheduler.step()
        eval_outputs = eval_loop(model, evaluate_attn, optimizer, lr_scheduler, train_config, fsdp_config, rank,
                                 eval_dataloader, local_rank, tokenizer, wandb_run,
                                 val_step_loss, val_loss, best_val_loss, checkpoint_times, 
                                 epoch, step)
        save_path, val_step_loss, val_loss, best_val_loss, checkpoint_times = eval_outputs
        if save_path is not None:
            best_checkpoint_path = save_path

        if train_config.enable_fsdp:
            if rank==0:
                print(f"Epoch {epoch+1}:  train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch+1}: train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        
        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, val_step_loss, val_loss)

    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename

    #saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp and not train_config.use_peft:
        save_train_params(train_config, fsdp_config, rank)

    return results, best_checkpoint_path


def eval_loop(model, evaluate_func, optimizer, lr_scheduler,
              train_config, fsdp_config, rank, eval_dataloader,
              local_rank, tokenizer, wandb_run,
              val_step_loss, val_loss, best_val_loss,
              checkpoint_times, epoch, step):  # extra globals
    """
    Evaluate model and save checkpoints
    - see `evaluate_func` for evaluation logic
    """
    eval_epoch_loss, temp_val_loss = evaluate_func(
        model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run, epoch=epoch)
    try:
        lr_scheduler.step(eval_epoch_loss)
    except:
        lr_scheduler.step()

    if train_config.save_metrics:
        val_step_loss.extend(temp_val_loss)

    checkpoint_start_time = time.perf_counter()

    # train_config.save_model = False
    if train_config.save_model and eval_epoch_loss < best_val_loss:
        if train_config.enable_fsdp:
            dist.barrier()
        else:
            dist.barrier()
            
        if fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
            save_path = save_model_checkpoint(
                model, optimizer, rank, train_config, epoch=epoch
            )
        elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
            if rank == 0:
                print("Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                print("==========================================================")

            save_path = save_model_and_optimizer_sharded(model, rank, train_config)
            if train_config.save_optimizer:
                save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                if rank == 0:
                    print("Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                    print("========================================================================")

        if train_config.save_optimizer:
            save_optimizer_checkpoint(
                model, optimizer, rank, train_config, epoch=epoch
            )
            if rank == 0:
                print("Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                print("=====================================================================")
        if train_config.enable_fsdp:
            dist.barrier()
        else:
            dist.barrier()
    else:
        save_path = None
    checkpoint_end_time = time.perf_counter() - checkpoint_start_time
    checkpoint_times.append(checkpoint_end_time)
    if eval_epoch_loss < best_val_loss:
        best_val_loss = eval_epoch_loss
        if rank == 0 or not train_config.enable_fsdp:
            print(f"best eval loss on epoch {epoch+1}, step {step + 1} is {best_val_loss}")

    val_loss.append(float(best_val_loss))
    return save_path, val_step_loss, val_loss, best_val_loss, checkpoint_times


def evaluate_attn(model, train_config, eval_dataloader,
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
            loss, loss_metrics = loss_computer.compute_loss(model, batch)
            if train_config.save_metrics:
                val_step_loss.append(loss.detach().float().item()) 

            eval_loss += loss.detach().float()

        desc = f"Rank {rank} | Eval Epoch{_epoch} | step_loss: {loss.item():.5f} | avg_loss: {eval_loss.item()/(step+1):.5f}"
        for k, v in loss_metrics.items():
            desc += f" | {k}: {v:.5f}"
        pbar.set_description(desc)

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss
    # print('len(eval_dataloader):', len(eval_dataloader))
    # print('step + 1:', step + 1)
    # print('world_size:', world_size)
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size

    eval_epoch_loss = eval_epoch_loss.cpu().float().item()

    # Print evaluation metrics
    if local_rank == 0 or not train_config.enable_fsdp:
        print(f" {eval_epoch_loss=}")

    if wandb_run:
        wandb_run.log({'eval/loss': eval_epoch_loss,}, commit=False)

    del loss; del eval_loss; del batch
    clear_gpu_cache()

    return eval_epoch_loss, val_step_loss


def freeze_transformer_layers(model: nn.Module, num_layer: int):
    """Freeze model layers up to num_layer"""
    for i, layer in enumerate(model.model.layers):
        if i < num_layer:
            for param in layer.parameters():
                param.requires_grad = False


def check_frozen_layers_peft_model(model) -> None:
    """
    Print layer.requires_grad for each layer in the model
    """
    for i, layer in enumerate(model.base_model.model.model.layers):
        for name, param in layer.named_parameters():
            print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup():
    """Initialize the process group for distributed training"""
    if is_ccl_available():
        # distributed training on xpus
        dist.init_process_group("ccl")
    else:
        dist.init_process_group("nccl", timeout=timedelta(seconds=3600))


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nightlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        print("--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print("Clearing GPU cache for all ranks")
    if is_xpu_available():
        torch.xpu_empty_cache()
    else:
        torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes


def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")


def get_policies(cfg, rank, model: str = 'llama'):
    """Get the policies for mixed precision and fsdp wrapping"""
    verify_bfloat_support = ((
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    ) or
    (is_xpu_available()))

    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print("bfloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print("FP16 enabled")
        else:
            print("bfloat16 support not present. Using FP32, and not mixed precision")

    if model == 'llama':
        wrapping_policy = get_llama_wrapper()
    elif model == 'mistral':
        wrapping_policy = get_mistral_wrapper()
    elif model == 'mixtral':
        wrapping_policy = get_mixtral_wrapper()
    return mixed_precision_policy, wrapping_policy


def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
    train_config.dist_checkpoint_root_folder
    + "/"
    + train_config.dist_checkpoint_folder
    + "-"
    + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            print(f"training params are saved in {file_name}")


def save_to_json(output_filename,
                 train_step_loss, train_epoch_loss,
                 val_step_loss, val_epoch_loss):
    """Save loss data to JSON file"""
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)