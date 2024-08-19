"""
Custom trainer class for training models with "chunked" linear attention

Lets us train like an RNN to process long sequences with fixed memory:
- Use linear attention's recurrent view to process a long sequence 
  as a set of non-overlapping chunks 
- At the end of each chunk, pass the computed KV and K states to initialize 
  the states for the next chunk
- Accumulate gradients with loss over each chunk
"""
import torch
import torch.nn as nn

from tqdm import tqdm

from src.model.modeling_llama import get_attention_cache
from src.model.convert_model import traverse_layers

from .default_lm import OurTrainer as OurDefaultTrainer
from .utils import decode_samples


class OurTrainer(OurDefaultTrainer):
    """
    Custom trainer class for training models with "chunked" linear attention

    Lets us train like an RNN to process long sequences with fixed memory:
    - Use linear attention's recurrent view to process a long sequence 
      as a set of non-overlapping chunks 
    - At the end of each chunk, pass the computed KV and K states to initialize 
      the states for the next chunk
    - Accumulate gradients with loss over each chunk
    """
    def __init__(self, model, **kwargs: any):
        assert (
            getattr(model, 'state_chunk_len', None) is not None
        ), "model must have a `state_chunk_len` attribute"
        super().__init__(model=model, **kwargs)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.tokenizer = getattr(self.train_loader.dataset, 'tokenizer', None)
        self.compute_loss_backprop = True  # Whether we backprop in self.compute_loss
        self.initial_eval = False  # Whether to evaluate before training

    def compute_loss(self, model: nn.Module, data: dict[torch.Tensor],
                     sample_idx: int = None, **kwargs: any,
                     ) -> tuple[torch.Tensor, dict[any]]:
        """
        Compute loss over sample as a sequence of chunks
        - model should have a `state_chunk_len` attribute and a `chunk_forward`
          method, where `state_chunk_len` defines the chunk size

        Args:
        - model: nn.Module, HF model to train
        - data: dict[torch.Tensor], HF datasets batch of data
        - sample_idx: int, index of batch in dataset
        """
        chunk_metrics = {}
        input_seq_len = data['input_ids'].shape[-1]
        total_seq_len = 0
        loss = 0
        
        # Get KV state object for model as `past_key_values`
        layers = traverse_layers(model)
        attention_type = getattr(layers[0].self_attn, 'attention_type', None)
        past_key_values = get_attention_cache(attention_type)
        
        # Chunk original input sequences; assume single batch for now
        input_ids = torch.split(data['input_ids'], model.state_chunk_len, dim=-1)
        labels = data['labels'] if 'labels' in data else data['input_ids']
        labels = torch.split(labels, model.state_chunk_len, dim=-1)
        
        pbar = tqdm(input_ids, desc=f'Processing chunk 0 | state len: {model.state_chunk_len} (token {total_seq_len} / {input_seq_len})', leave=False)
        
        for chunk_idx, chunk_input_ids in enumerate(pbar):
            try:
                outputs = model.chunk_forward(input_ids=chunk_input_ids.to(self.device),
                                              attention_mask=None,
                                              position_ids=None,
                                              past_key_values=past_key_values,
                                              inputs_embeds=None,
                                              use_cache=True,
                                              output_attentions=False,
                                              output_hidden_states=False,
                                              return_dict=True)
            except Exception as e:
                raise e
            
            past_key_values = outputs.past_key_values
            outputs = outputs.get('logits')[..., :-1, :].contiguous()
            targets = labels[chunk_idx][..., 1:].contiguous()

            if ((targets != -100).sum() > 0 and self.tokenizer is not None and
                sample_idx is not None and (sample_idx + 1) % 100 == 0):
                decode_samples(outputs.cpu(), targets.cpu(), self.tokenizer, sample_idx)
            
            outputs = outputs.view(-1, outputs.shape[-1])
            targets = targets.view(-1).to(outputs.device)

            if (targets != -100).sum() == 0:  # Chunk contains only padding or prompts
                chunk_metrics[f'loss_{chunk_idx}'] = 0
                chunk_metrics[f'ppl_{chunk_idx}'] = 1  # torch.exp(_loss).item(), or -1?
            else:
                try:
                    _loss = self.criterion(outputs, targets)
                except Exception as e:
                    print(e)
                    breakpoint()

                # Accumulate gradients over chunks    
                if model.training:
                    with torch.autograd.set_detect_anomaly(True):
                        _loss.backward()
                        
                chunk_metrics[f'loss_{chunk_idx}'] = _loss.item()
                chunk_metrics[f'ppl_{chunk_idx}'] = torch.exp(_loss).item()
            loss += chunk_metrics[f'loss_{chunk_idx}'] / len(input_ids)
            total_seq_len += chunk_input_ids.shape[-1]
            desc=f'Processing chunk {chunk_idx + 1} | state len: {model.state_chunk_len} (token {total_seq_len} / {input_seq_len}) | loss: {chunk_metrics[f"loss_{chunk_idx}"]:.3f} | ppl: {chunk_metrics[f"ppl_{chunk_idx}"]:.3f}'
            pbar.set_description(desc)
            
            targets = targets.cpu()
            outputs = outputs.cpu()
            _loss = _loss.cpu()
        del past_key_values, outputs, targets, _loss
        torch.cuda.empty_cache()

        # Display chunks in reverse
        chunk_metrics = [(k, v) for k, v in chunk_metrics.items()][::-1]
        chunk_metrics = {k: v for k, v in chunk_metrics}
        return loss, chunk_metrics
