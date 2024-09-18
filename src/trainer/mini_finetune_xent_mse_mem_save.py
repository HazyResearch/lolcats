"""
Custom trainer class for distilling attentions. Can substitute for HuggingFace trainer.
"""
import torch
import torch.nn as nn

from .default_lm import OurTrainer as DefaultTrainer

from src.model.convert_model import traverse_layers, toggle_attention
from peft.tuners.lora.layer import LoraLayer

def toggle_lora(model, use_lora: bool = True):
    for layer in traverse_layers(model):
        for n, module in layer.self_attn.named_modules():
            if isinstance(module, LoraLayer):
                module._disable_adapters = not use_lora
    return model

class OurTrainer(DefaultTrainer):
    """
    This one is to help us just have one model stored in GPU memory -- nice for 405B.

    Custom trainer class for distilling attentions. 
    - We compute and store the attention outputs and/or weights for each head and layer,
      for both the "teacher" softmax attentions and "student" learnable subquadratic attentions
    - We then train the student layers to minimize either MSE(outputs) or CrossEntropy(weights)
    """
    def __init__(self,
                 model: nn.Module,
                 layer_idx: int,
                 metric_for_best_model: str = 'ft/eval/loss',
                 mse_factor: float = 1e3,
                 xent_factor: float = 0,
                 **kwargs: any):
        model = toggle_attention(model, train=True)  # keep train_attention logic
        super().__init__(model=model, 
                         metric_for_best_model=metric_for_best_model,
                         **kwargs)
        self.criterion_xent = nn.CrossEntropyLoss(reduction='mean')  # stability
        self.criterion_mse = nn.MSELoss(reduction='mean')
        self.mse_factor = mse_factor
        self.xent_factor = xent_factor
        self.layer_idx = layer_idx
        self.initial_eval = False  # Whether to evaluate before training

    def compute_loss(self, model: nn.Module, data: dict[torch.Tensor],
                     sample_idx: int = None, **kwargs: any,) -> tuple[torch.Tensor, dict[any]]:
        """
        Attention distillation ("attention transfer")
        - For each layer and head, get attentions and train to 
          minimize some combo of MSE and cross-entropy loss

        model: nn.Module that is a Lolcats attention class. 
               If outputs = model(**inputs), 
               - outputs[0] are the layer outputs
               - outputs[1] are attentions (or other saved tensors)
        """
        try:
            _data_kwargs = {'device': model.q_proj.weight.device, 
                            'dtype':  model.q_proj.weight.dtype}
        except:
            ref_weight = model.model.model.layers[0].self_attn.q_proj.base_layer.weight
            _data_kwargs = {'device': ref_weight.device, 
                            'dtype':  ref_weight.dtype}
        
        inputs = {'inputs_embeds': data['inputs_embeds'].to(**_data_kwargs)}
        if 'position_ids' in data:
            inputs['position_ids'] = data['position_ids'].to(**_data_kwargs)

        # Teacher outputs
        with torch.no_grad():
            model = toggle_lora(model, use_lora=False)
            outputs = model(**inputs, output_attentions=True, use_cache=False)
            outputs = outputs.get('attentions')  
            a_true = [o[0][1] for o in outputs]
            y_true = [o[1][1] for o in outputs]

        # Student outputs
        model = toggle_lora(model, use_lora=True)
        outputs = model(**inputs, output_attentions=True, use_cache=False).get('attentions')
        a_pred = [o[0][0] for o in outputs]
        y_pred = [o[1][0] for o in outputs]
    
        inputs = {k: v.cpu() for k, v in inputs.items()}  # save gpu memory

        # Initialize variables to track average magnitudes
        avg_attn_mag_pred, avg_attn_mag_true = 0, 0 
        avg_output_mag_pred, avg_output_mag_true = 0, 0
        attention_count = 0
        output_count = 0

        loss_mse = 0
        loss_xent = 0
        skip_xent_mem_save = False
        for layer_idx in range(len(a_pred)):  # indexed by n_layers
            if self.xent_factor > 0:
                _a_pred, _a_true = a_pred[layer_idx], a_true[layer_idx]
                if _a_pred is not None:
                
                    # Cross-entropy loss
                    _a_pred = _a_pred.clamp(min=1e-12).log()  # nn.CrossEntropy assumes unnormalized logits
                    k_len = _a_true.shape[-1]  # batch, n_heads, q_len, k_len
                    
                    # Compute mean cross-entropy over all queries
                    _a_pred = _a_pred.contiguous().view(-1, k_len)
                    _a_true = _a_true.contiguous().view(-1, k_len)
                    loss_xent += self.criterion_xent(_a_pred, _a_true)

                    # Calculate and accumulate average magnitude for attention
                    avg_attn_mag_pred += torch.abs(_a_pred).mean().item()
                    avg_attn_mag_true += torch.abs(_a_true).mean().item()
                    attention_count += 1
                else:
                    skip_xent_mem_save = True
            else:
                skip_xent_mem_save = True
                
            if self.mse_factor > 0:
                loss_mse += self.criterion_mse(y_pred[layer_idx], y_true[layer_idx])

                # Calculate and accumulate average magnitude for outputs
                avg_output_mag_pred += torch.abs(y_pred[layer_idx]).mean().item()
                avg_output_mag_true += torch.abs(y_true[layer_idx]).mean().item()
                output_count += 1

        loss_xent = loss_xent / len(y_pred) * self.xent_factor
        loss_mse = loss_mse / len(y_pred) * self.mse_factor

        # Calculate final average magnitudes
        avg_attn_mag_pred = avg_attn_mag_pred / attention_count if attention_count > 0 else 0
        avg_output_mag_pred = avg_output_mag_pred / output_count if output_count > 0 else 0
        avg_attn_mag_true = avg_attn_mag_true / attention_count if attention_count > 0 else 0
        avg_output_mag_true = avg_output_mag_true / output_count if output_count > 0 else 0
        
        if skip_xent_mem_save:
            loss = loss_mse
        else:
            loss = loss_xent + loss_mse

        outputs = {'loss_xent': loss_xent.item() if self.xent_factor > 0 else 0,
                   'loss_mse': loss_mse.item() if self.mse_factor > 0 else 0, 
                   'mse_factor': self.mse_factor, 
                   'xent_factor': self.xent_factor,
                   'layer_idx': self.layer_idx,
                   'avg_attn_mag_pred': avg_attn_mag_pred,
                   'avg_output_mag_pred': avg_output_mag_pred,
                   'avg_attn_mag_true': avg_attn_mag_true,
                   'avg_output_mag_true': avg_output_mag_true,
                }
        return loss, outputs
    

