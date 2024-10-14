"""
Default trainer class for training models
"""
from collections import OrderedDict
from os.path import join
from argparse import ArgumentParser
from tqdm import tqdm

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .optim import get_optimizer, get_scheduler
from .utils import decode_samples


class OurTrainer():
    """
    Basic parent trainer class. Defaults to language modeling. 
    -> Replacement for Hugging Face Trainer
    """
    def __init__(self,
                 model: nn.Module,
                 args: ArgumentParser,
                 train_loader: DataLoader,
                 eval_loader: DataLoader,
                 optimizer_and_scheduler: tuple[Optimizer, LRScheduler],
                 device: torch.device,
                 wandb,  # WandB object
                 checkpoint_suffix: str = None,
                 save_checkpoints: bool = True,
                 save_results: bool = True,
                 # Custom arguments
                 optimizer_args: dict = None,
                 lr_scheduler_args: dict = None,
                 greater_is_better: bool = False,
                 metric_for_best_model: str = 'eval/loss',
                 num_train_epochs: int = 2,
                 gradient_accumulation_steps: int = 1,
                 evaluation_strategy: str = 'steps',
                 load_best_model_at_end: bool = True,
                 logging_steps: int = 100,
                 max_steps: int = -1,
                 eval_steps: int = 100,
                 max_eval_batches: int = -1,
                 print_samples: bool = False,
                 initial_eval: bool = True,
                 num_save_ckpt_steps: int = 1000,
                 **kwargs: any):
        super().__init__()
        self.model = model
        self.step = 0  # Total steps taken
        self.grad_step = 0  # Total gradient updates
        self.compute_loss_backprop = False  # Whether we backprop in self.compute_loss

        if optimizer_and_scheduler is None:
            assert optimizer_args is not None and lr_scheduler_args is not None
            self.optimizer = get_optimizer(model=self.model, **optimizer_args)
            self.scheduler = get_scheduler(optimizer=self.optimizer, **lr_scheduler_args)
        else:
            self.optimizer, self.scheduler = optimizer_and_scheduler
        try:
            self.scheduler_step_after_epoch = 'plateau' in args.lr_scheduler['lr_scheduler_type']
        except KeyError:
            self.scheduler_step_after_epoch = False

        # Dataloaders
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.device = device
        self.wandb = wandb
        
        # Custom arguments
        self.metric_for_best_model = metric_for_best_model
        self.num_train_epochs = num_train_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.evaluation_strategy = evaluation_strategy
        self.greater_is_better = greater_is_better
        self.is_better = (lambda x, y: x > y if greater_is_better else x < y)
        self.load_best_model_at_end = load_best_model_at_end
        self.logging_steps = logging_steps
        self.max_steps = max_steps
        self.eval_steps = eval_steps
        self.max_eval_batches = max_eval_batches
        self.print_samples = print_samples
        self.initial_eval = initial_eval
        self.num_save_ckpt_steps = num_save_ckpt_steps

        # Saving metrics
        self.train_metrics = {'train/loss': None, 
                              'train/epoch': None, 
                              'train/step': None}
        self.eval_metrics = {metric_for_best_model: None}
        self.eval_metrics_by_step = {'eval_step': []}  # save all eval metrics
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        try:
            self.tokenizer = self.train_loader.dataset.tokenizer
        except AttributeError:
            self.tokenizer = None
            
        self.save_results = save_results
        self.results_path = None
        self.best_val_metric = 0 if greater_is_better else 1e10
        self.best_val_metric_epoch = 0
        self.best_val_metric_step = 0
        if save_checkpoints:  # Also initializes best_val_metrics
            self.init_checkpointing(args=args, checkpoint_suffix=checkpoint_suffix)

    def train(self) -> nn.Module:
        """
        Entire training run
        """
        model = self.model
        pbar = tqdm(range(self.num_train_epochs), leave=False, colour='white',
                    desc='Training')
        for ix, epoch in enumerate(pbar):
            model, early_stopping = self.train_step(model, epoch)
            if self.evaluation_strategy == 'epoch':
                _eval_metrics = self.eval_step(model, step=self.grad_step)
                print(f'Epoch {ix} metrics:', _eval_metrics)
            if early_stopping:
                break
                
        if self.load_best_model_at_end:  # Return best checkpoint
            try:
                state_dict = torch.load(self.best_val_checkpoint_path)['model_state_dict']
                model.load_state_dict(state_dict, strict=False)
                print(f'-> Loading best checkpoint from {self.best_val_checkpoint_path}')
            except FileNotFoundError as e:
                print(e)
                print('-> Returning most recent model instead')
        return model            

    def train_step(self, model: nn.Module, epoch: int) -> nn.Module:
        """
        Training loop over one epoch
        """
        if self.gradient_accumulation_steps is None:
            accum_iter = 1
        else:
            accum_iter = self.gradient_accumulation_steps

        model.train()
        model.zero_grad()        
        pbar = tqdm(self.train_loader, leave=False, colour='blue',
                    desc=f'-> Training (epoch {epoch} / {self.num_train_epochs})')
        total_loss = 0
        eval_for_step = False

        # Initial eval
        if self.initial_eval:
            print('')
            print('-> Initial eval')
            self.compute_eval_metrics(model, step=self.grad_step)
        
        # model.to(self.device)
        for ix, data in enumerate(pbar):
            loss, train_metrics = self.compute_loss(model, data, 
                                                    sample_idx=ix)
            loss /= accum_iter
            if not self.compute_loss_backprop:
                # loss.backward() did not occur in compute_loss
                try:
                    with torch.autograd.set_detect_anomaly(True):
                        loss.backward()
                except Exception as e:
                    breakpoint()
            if (self.step + 1) % accum_iter == 0:  # and self.step != 0:
                self.optimizer.step()
                if not self.scheduler_step_after_epoch and self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self.grad_step += 1
                if not self.compute_loss_backprop:
                    loss = loss.detach().cpu().item()

            self.step += 1
            if not isinstance(loss, float):
                total_loss += loss.item()
            else:
                total_loss += loss
            desc = f"Training epoch {epoch} | loss: {total_loss / (ix + 1):.3f} | lr: {self.optimizer.param_groups[0]['lr']:.5f}"
            desc += f' | gradient step: {self.grad_step}'
            for k, v in train_metrics.items():
                desc += f' | {k}: {v:.3f}'
            pbar.set_description(desc)

            # Logging
            if (self.grad_step) % (self.logging_steps):
                self.train_metrics['train/loss'] = loss.item() if not isinstance(loss, float) else loss
                self.train_metrics['train/epoch'] = epoch
                self.train_metrics['train/step'] = self.grad_step
                self.train_metrics['train/lr'] = self.optimizer.param_groups[0]['lr']
                for k, v in train_metrics.items():
                    self.train_metrics[f'train/{k}'] = v
                
                if self.wandb is not None:
                    self.wandb.log(self.train_metrics, step=self.grad_step)

            if self.evaluation_strategy == 'steps':
                if (self.grad_step % self.eval_steps == 0 and self.grad_step > 0 and not eval_for_step):
                    _eval_metrics = self.eval_step(model, step=self.grad_step)
                    print(f'Grad Step {self.grad_step} eval metrics:', _eval_metrics)
                    eval_for_step = True
                    model.train()  # Need to set back to train mode
                elif self.grad_step == 0 and self.num_save_ckpt_steps < 1000 and not eval_for_step:  # hack for micros
                    _eval_metrics = self.eval_step(model, step=self.grad_step)
                    print(f'Grad Step {self.grad_step} eval metrics:', _eval_metrics)
                    eval_for_step = True
                    model.train()  # Need to set back to train mode
                    
                elif self.grad_step % self.eval_steps == 0 and self.grad_step > 0 and eval_for_step:
                    pass
                else:
                    if self.grad_step > 0:
                        eval_for_step = False
            if self.grad_step == self.max_steps:
                early_stopping = True
                return model, early_stopping
                
        early_stopping = False
        return model, early_stopping

    def eval_step(self, model: nn.Module, step: int = None,
                  **kwargs: any) -> dict[any]:
        """
        Evaluation loop over one epoch
        """
        with torch.no_grad():
            self.eval_metrics = self.compute_eval_metrics(model, step=step, **kwargs)
            val_metric = self.eval_metrics[self.metric_for_best_model]

            # Save results
            if self.wandb is not None:  # log to WandB
                self.wandb.log(self.eval_metrics, step=self.grad_step)

            if self.results_path is not None:  # log to local file
                self.eval_metrics_by_step['eval_step'].append(step)
                for k, v in self.eval_metrics.items():
                    if k not in self.eval_metrics_by_step:
                        self.eval_metrics_by_step[k] = [v]
                    else:
                        self.eval_metrics_by_step[k].append(v)
                # Inefficient, but log for experiments results
                pd.DataFrame(self.eval_metrics_by_step).to_csv(self.results_path)

            # Save best metric and checkpoint
            if self.grad_step % self.eval_steps == 0:
                if self.is_better(val_metric, self.best_val_metric):
                    self.best_val_metric = val_metric
                    self.best_val_metric_step = self.grad_step
                    # model.cpu()
                    torch.save({
                        'model_state_dict': self.save_trainable_weights(model),
                        'step': self.grad_step, 
                        self.metric_for_best_model: val_metric
                    }, self.best_val_checkpoint_path)
                    print(f'\n-> Saved best model checkpoint to: {self.best_val_checkpoint_path}!')

            if self.grad_step % self.num_save_ckpt_steps == 0:
                save_path = self.best_val_checkpoint_path.replace('.pt', f'_{self.grad_step}.pt')
                torch.save({
                    'model_state_dict': self.save_trainable_weights(model),
                    'step': self.grad_step, 
                    self.metric_for_best_model: val_metric
                }, save_path)
                print(f'\n-> Saved best model checkpoint to: {save_path}!')
            
            if self.scheduler_step_after_epoch and self.scheduler is not None:
                self.scheduler.step(val_metric)
            return self.eval_metrics

    def compute_eval_metrics(self, 
                             model: nn.Module, step: int,
                             max_batches: int = None,
                             dataloader: DataLoader = None,
                             **kwargs: any) -> dict[any]:
        """
        One evaluation loop over a validation dataset
        """
        max_batches = (self.max_eval_batches if max_batches is None else max_batches)
        dataloader = self.eval_loader if dataloader is None else dataloader
        pbar = tqdm(dataloader, leave=False, colour='green',
                    desc=f'Evaluating at step {step}')
        
        model.eval()
        step_loss = 0
        step_eval_metrics = {}
        with torch.no_grad():
            for ix, data in enumerate(pbar):
                loss, eval_metrics = self.compute_loss(model, data)
                if not self.compute_loss_backprop:
                    loss = loss.item()  # otherwise already float
                if ix == 0:
                    step_eval_metrics[self.metric_for_best_model] = [loss]
                    for k, v in eval_metrics.items():
                        step_eval_metrics[f'eval/{k}'] = [v]
                else:
                    step_eval_metrics[self.metric_for_best_model].append(loss)
                    for k, v in eval_metrics.items():
                        step_eval_metrics[f'eval/{k}'].append(v)
                        
                step_loss += loss
                desc = f"Evaluating at step {step} | loss: {step_loss / (ix + 1):.3f}"
                if self.optimizer is not None:
                    desc += f" | lr: {self.optimizer.param_groups[0]['lr']:.5f}"
                pbar.set_description(desc)
                if ix == max_batches:
                    break

            # Average over batches
            for k, v in step_eval_metrics.items():
                step_eval_metrics[k] = sum(v) / len(v)
            print(f'Eval step {step}:', step_eval_metrics)
            del loss
            torch.cuda.empty_cache()
        return step_eval_metrics

    def compute_loss(self, model: nn.Module, data: torch.Tensor,
                     sample_idx: int = None, **kwargs: any,
                     ) -> tuple[torch.Tensor, dict[any]]:
        """
        Main method to determine how models are trained. 
        -> Defaults to next-token prediction / classification, 
           but override in child classes

        Args:
        - model: nn.Module, HF model to train
        - data: dict[torch.Tensor], HF datasets batch of data
        - sample_idx: int, index of batch in dataset
        """
        input_keys = {'input_ids', 'attention_mask'}
        inputs = {k: v.to(model.device) 
                  for k, v in data.items() if k in input_keys}  

        outputs = model(**inputs, output_attentions=False, use_cache=False)
        
        outputs = outputs.get('logits')[..., :-1, :].contiguous()
        targets = data.get('labels')[..., 1:].contiguous()

        # Look at model outputs
        if self.print_samples and sample_idx is not None and (sample_idx + 1) % 100 == 0:
            decode_samples(outputs, targets, self.tokenizer, sample_idx)

        # Flatten and compute cross-entropy loss
        outputs = outputs.view(-1, outputs.shape[-1])
        targets = targets.view(-1).to(outputs.device)
        try:
            loss = self.criterion(outputs, targets)
        except Exception as e:
            print('outputs.shape', outputs.shape)
            print('targets.shape', targets.shape)
            raise e

        targets = targets.cpu()
        outputs = outputs.cpu()
        return loss, {'ppl': torch.exp(loss).item(), 'seq_len': targets.shape[-1] + 1}

    def save_trainable_weights(self, model: nn.Module):
        """
        Save checkpoint with only weights actively being trained (e.g., for adapters). 
        Make sure to later load with model.load_state_dict(state_dict, strict=False)
        """
        with torch.no_grad():
            state_dict = OrderedDict()
            for n, p in model.named_parameters():
                if p.requires_grad:
                    state_dict[n] = p.cpu()  # assurance
            return state_dict
        
    def init_checkpointing(self,
                           args: ArgumentParser,
                           checkpoint_suffix: str) -> None:
        """
        Initialize checkpointing attributes

        Inputs:
        - args: Argparse or HuggingFace TrainingArguments object
        - checkpoint_suffix: str to append to checkpoint name
        """
        self.best_val_checkpoint_path = f'{join(args.checkpoint_dir, args.run_name)}.pt'
        if checkpoint_suffix is not None:
            self.best_val_checkpoint_path = self.best_val_checkpoint_path.replace(
                '.pt', f'{checkpoint_suffix}.pt')
        print(f'-> Saving best model checkpoint to {self.best_val_checkpoint_path}')
        if self.save_results:
            self.results_path = self.best_val_checkpoint_path.replace(
                '.pt', '.csv').replace(args.checkpoint_dir, args.results_dir)
            print(f'-> Saving results to {self.results_path}')

        # Best metric setup
        self.best_val_metric = 0 if self.greater_is_better else 1e10
        self.best_val_metric_epoch = 0
        self.best_val_metric_step = 0
        self.best_train_metric = 0 if self.greater_is_better else 1e10
        self.best_train_metric_epoch = 0
        self.best_train_metric_step = 0
        self.metric_for_best_model = self.metric_for_best_model
        if self.metric_for_best_model is not None:
            if 'eval' not in self.metric_for_best_model:
                self.metric_for_best_model = f'eval/{self.metric_for_best_model}'
