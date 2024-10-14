"""
General seq2seq / input-output trainer
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from .default_lm import OurTrainer as DefaultTrainer
from .utils import replace_padding_tokens


def compute_scrolls_metrics(eval_preds, scrolls_metric, tokenizer):
    """
    Function to compute metrics that are also in SCROLLS (ROUGE, F1, etc.)
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100s used for padding as we can't decode them
    preds  = replace_padding_tokens(preds, tokenizer.pad_token_id)
    labels = replace_padding_tokens(labels, tokenizer.pad_token_id)

    decoded_preds  = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Scrolls metric expects predictions to be [pred_1, pred_2, ...]
    # and references to be [[ref_1], [ref_2], ... ]
    decoded_labels = [[s] for s in decoded_labels]
    
    result = scrolls_metric.compute(predictions=decoded_preds, 
                                    references=decoded_labels)
    print('----------------')
    print('Model generation')
    print(decoded_preds[:10])
    print('----------------')
    print('True answer')
    print(decoded_labels[:10])
    return result


class OurTrainer(DefaultTrainer):
    """
    Evaluator for seq-to-seq / generation benchmarks
    """
    def __init__(self, model, args, # max_eval_batches: Optional[int] = 100,
                 **kwargs: any):
        super().__init__(model=model, args=args, **kwargs)
        # Reset + determine metric for best automatically based on the dataset
        self.metric_for_best = None
        self.is_better = lambda x, y: x > y  # Hardcode greater is better for now
        self.print_steps = getattr(args, 'print_steps', 100)
        print(f'self.print_steps:', self.print_steps)
        # ablation sweep
        self.max_eval_batches = 10

    def init_criterion_(self):
        pass

    def compute_loss(self):
        pass

    def evaluate(self, *args: any, **kwargs: any):
        return self.eval_step(*args, **kwargs)

    def eval_step(self, model: nn.Module, step: int, 
                  dataloader: DataLoader = None,
                  max_batches: int = None,
                  prefix: str = None,
                  **kwargs: any):  # -1):
        """
        One evaluation step
        """
        total = 0
        total_loss = 0
        metrics = {}
        max_batches = self.max_eval_batches if max_batches is None else max_batches
        max_batches = 10  # ablation sweep

        dataloader = (dataloader if dataloader is not None else self.eval_loader)

        scrolls_metric = dataloader.dataset.metric  # Should be assigned in dataset
        tokenizer = dataloader.dataset.tokenizer

        # Save decoded predictions and references here to compute average metrics
        predictions, references = [], []

        model.eval()
        
        pbar = tqdm(dataloader, leave=False, colour='green', 
                    desc=f'Evaluating at step {step}')
        
        with torch.no_grad():
            for ix, data in enumerate(pbar):
                inputs = {k: v.to(self.device) for k, v in data.items() 
                          if k in ['input_ids', 'attention_mask']}
                labels = data['labels']
                outputs = model.generate(**inputs,
                                         max_new_tokens=1024,  # hardcoded for now
                                         pad_token_id=tokenizer.pad_token_id,
                                         use_cache=True,).cpu()
                # Only save newly generated tokens
                pred_ids = outputs[:, data['input_ids'].shape[1]:]
                predictions.append(pred_ids)
                references.append(labels)
                pbar.set_description(f"Evaluating at step {step} | input_len: {data['input_ids'].shape[1]} | output_len: {labels.shape[1]}")

                if ix == max_batches:
                    break

                if (ix + 1) % self.print_steps == 0:  # 100 == 0:
                    print(f'Model input: \n', tokenizer.batch_decode(inputs['input_ids'].detach().cpu())[0])
                    print(f'Model output:\n', tokenizer.batch_decode(pred_ids)[0])
                    print(f'True  output:\n', tokenizer.batch_decode(labels)[0])

            # Compute and save metrics
            try:
                predictions = torch.cat(predictions, dim=0)
                references  = torch.cat(references, dim=0)
            except:
                pass
            _metric = compute_scrolls_metrics((predictions, references),
                                              scrolls_metric, tokenizer)
            if self.metric_for_best is None:  # Hard-coded for now
                if 'f1' in _metric:
                    self.metric_for_best = f'eval/f1'
                elif 'exact_match' in _metric:
                    self.metric_for_best = f'eval/exact_match'
                elif 'rouge/geometric_mean' in _metric:
                    self.metric_for_best = f'eval/rouge/geometric_mean'
            for k, v in _metric.items():
                if 'display' not in k:
                    _k = f'{prefix}/eval/{k}' if prefix is not None else f'eval/{k}'
                    metrics[_k] = v
            
        return metrics
