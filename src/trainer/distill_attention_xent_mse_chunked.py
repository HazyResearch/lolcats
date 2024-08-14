"""
Custom trainer class for distilling attentions over long sequences with 
recurrent linear attention view. Can substitute for HuggingFace trainer.
"""
import torch
import torch.nn as nn

from tqdm import tqdm

from src.model.modeling_llama import get_attention_cache
from src.model.convert_model import traverse_layers
from .default_lm_chunked import OurTrainer as DefaultChunkedTrainer


class OurTrainer(DefaultChunkedTrainer):
    """
    Custom trainer class for distilling attentions. 
    - We compute and store the attention outputs and/or weights for each head and layer,
      for both the "teacher" softmax attentions and "student" learnable subquadratic attentions
    - We then train the student layers to minimize either MSE(outputs) or CrossEntropy(weights)
    """
    def __init__(self,
                 model: nn.Module,
                 metric_for_best_model: str = 'distill/eval/loss',
                 mse_factor: float = 1e3,
                 xent_factor: float = 0,
                 **kwargs: any):
        super().__init__(model=model, 
                         metric_for_best_model=metric_for_best_model,
                         **kwargs)
        self.criterion_xent = nn.CrossEntropyLoss(reduction='mean')  # stability
        self.criterion_mse = nn.MSELoss(reduction='mean')
        self.mse_factor = mse_factor
        self.xent_factor = xent_factor
        self.compute_loss_backprop = True  # Whether we backprop in self.compute_loss


    def compute_loss(self, model: nn.Module, data: dict[torch.Tensor],
                     sample_idx: int = None, **kwargs: any,) -> tuple[torch.Tensor, dict[any]]:
        """
        Attention distillation ("attention transfer")
        - For each layer and head, get attentions and train to 
          minimize some combo of MSE and cross-entropy loss
        """
        input_seq_len = data['input_ids'].shape[-1]
        inputs = {'input_ids': data['input_ids'].to(model.device)}  # assume all inputs good

        # Get softmax attention outputs
        with torch.no_grad():
            # Set base_inference to True to use FlashAttention
            for layer in traverse_layers(model):
                layer.self_attn.base_inference = True
            true_outputs = model.chunk_forward(**inputs, output_attentions=True, 
                                               use_cache=False,)
                                            #    no_logit_float=True,)
            # Hack were we save attention layer inputs and outputs in outputs.attentions
            # -> see model/hedgehog_attention_tk_long.py
            # attn_inputs  = [a[0] for a in true_outputs.get('attentions')]
            # attn_outputs = [a[1] for a in true_outputs.get('attentions')]
            true_attn_io = true_outputs.get('attentions')  # layer-wise attn inputs and outputs
            true_outputs = true_outputs.get('logits').cpu()
            for layer in traverse_layers(model):
                layer.self_attn.base_inference = False
        inputs = {k: v.cpu() for k, v in inputs.items()}
        torch.cuda.empty_cache()

        # Get trainable subquadratic attention outputs
        attention_type = getattr(layer.self_attn, 'attention_type', None)
        past_key_values = get_attention_cache(attention_type)

        num_chunks = input_seq_len // model.state_chunk_len
        total_seq_len = 0
        pbar = tqdm(range(num_chunks), desc=f'Processing chunk 0 | state len: {model.state_chunk_len} (token {total_seq_len} / {input_seq_len})', leave=False)

        position_ids = torch.arange(input_seq_len).view(1, -1)

        total_loss = 0
        for chunk_idx in pbar:
            start, end  = chunk_idx * model.state_chunk_len, (chunk_idx+1) * model.state_chunk_len
            attn_inputs = [o[0][:, start:end] for o in true_attn_io]  
            attn_output = [o[1][:, start:end] for o in true_attn_io]

            # Supervise attentions
            pos_ids = position_ids[:, start:end]
            loss_mse = 0
            loss_xent = 0
            for layer_idx, layer in enumerate(traverse_layers(model)):
                attn_preds = layer.self_attn(attn_inputs[layer_idx].to(model.device),
                                             attention_mask=None,
                                             position_ids=pos_ids.to(model.device),
                                             past_key_value=past_key_values)
                (attn_preds, attn_weights), past_key_values = (
                    attn_preds[1], attn_preds[2]
                )
                if self.mse_factor > 0:
                    # MSE on layer outputs
                    loss_mse += self.criterion_mse(attn_preds, attn_output[layer_idx].to(model.device))

                if self.xent_factor > 0:
                    # Cross-entropy on attention weights
                    aw_pred, aw_true = attn_weights
                    k_len = aw_pred.shape[-1]
                    # Compute mean loss only over individual queries
                    aw_pred = aw_pred.contiguous().view(-1, k_len).clamp(min=1e-12).log()  
                    aw_true = aw_true.contiguous().view(-1, k_len)
                    loss_xent += self.criterion_xent(aw_pred, aw_true)

            loss_mse = loss_mse / (layer_idx + 1) * self.mse_factor
            loss_xent = loss_xent / (layer_idx + 1) * self.xent_factor

            loss = loss_mse + loss_xent
            if model.training:
                loss.backward()

            desc=f'Processing chunk {chunk_idx + 1} | state len: {model.state_chunk_len} (token {end} / {input_seq_len}) | loss: {loss.item():.3f}'
            if self.mse_factor > 0:
                desc += f' | mse: {loss_mse.item():.3f}'
            if self.xent_factor > 0:
                desc += f' | xent: {loss_xent.item():.3f}'
            pbar.set_description(desc)

            total_loss += loss.item() / len(pbar)
            pbar.set_description(desc)

        if self.mse_factor > 0 or self.xent_factor > 0:
            loss = loss.cpu()
            if self.xent_factor > 0:
                attn_preds = attn_preds.cpu()
            torch.cuda.empty_cache()

        if 'position_ids' in data:
            outputs = {'loss_mse': loss_mse.item() if self.mse_factor > 0 else 0,
                       'loss_xent': loss_xent.item() if self.xent_factor > 0 else 0,
                       'mse_factor': self.mse_factor, 
                       'xent_factor': self.xent_factor,
                       'input_len': data['position_ids'].shape[1],
                       'position_ids': data['position_ids'][0],}
        else:
            outputs = {'loss_mse': loss_mse.item() if self.mse_factor > 0 else 0,
                       'loss_xent': loss_xent.item() if self.xent_factor > 0 else 0,
                       'mse_factor': self.mse_factor,
                       'xent_factor': self.xent_factor,}
        return total_loss, outputs