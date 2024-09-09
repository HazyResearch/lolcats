"""
Custom trainer class for distilling attentions. Can substitute for HuggingFace trainer.
"""
import torch
import torch.nn as nn

from peft.tuners.lora.layer import LoraLayer

from .default_lm import OurTrainer as DefaultTrainer
from src.model.convert_model import traverse_layers, toggle_attention



def toggle_lora(model, use_lora: bool = True):
    for layer in traverse_layers(model):
        for n, module in layer.self_attn.named_modules():
            if isinstance(module, LoraLayer):
                module._disable_adapters = not use_lora
    return model


class OurTrainer(DefaultTrainer):
    """
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

        self._data_kwargs = {'device': model.device, 
                             'dtype': traverse_layers(model)[0].self_attn.q_proj.weight.dtype}

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
        # inputs = {'inputs_embeds': data['hidden_states'].to(**_data_kwargs)}
        inputs = {'inputs_embeds': data['inputs_embeds'].to(**self._data_kwargs)}
        if 'position_ids' in data:
            inputs['position_ids'] = data['position_ids'].to(device=self.data_kwargs['device'])

        # Teacher outputs
        with torch.no_grad():
            model = toggle_lora(model, use_lora=False)
            outputs = model(**inputs, output_attentions=True, use_cache=False)
            outputs = outputs.get('attentions')  # ((_, a_true), (_, _y_true)) x layers
            a_true = [o[0][1] for o in outputs]
            y_true = [o[1][1] for o in outputs]
            # y_true = self.teacher_layer(**inputs, output_attentions=True, output_hidden_states=True, use_cache=False)
            # y_true = model(**inputs, output_attentions=True, output_hidden_states=True, use_cache=False)
            # y_true, a_true = y_true.get('hidden_states'), y_true.get('attentions')

        # Student outputs
        model = toggle_lora(model, use_lora=True)
        outputs = model(**inputs, output_attentions=True, use_cache=False).get('attentions')
        a_pred = [o[0][0] for o in outputs]
        y_pred = [o[1][0] for o in outputs]

        # y_pred = model(**inputs, output_attentions=True, output_hidden_states=True, use_cache=False)
        # y_pred, a_pred = y_pred.get('hidden_states'), y_pred.get('attentions')
    
        inputs = {k: v.cpu() for k, v in inputs.items()}  # save gpu memory

        loss_mse = 0
        loss_xent = 0
        for layer_idx in range(len(a_pred)):  # indexed by n_layers
            if self.xent_factor > 0:
                _a_pred, _a_true = a_pred[layer_idx], a_true[layer_idx]
                
                # Cross-entropy loss
                _a_pred = _a_pred.clamp(min=1e-12).log()  # nn.CrossEntropy assumes unnormalized logits
                k_len = _a_true.shape[-1]  # batch, n_heads, q_len, k_len
                
                # Compute mean cross-entropy over all queries
                _a_pred = _a_pred.contiguous().view(-1, k_len)
                _a_true = _a_true.contiguous().view(-1, k_len)
                loss_xent += self.criterion_xent(_a_pred, _a_true)
                
            if self.mse_factor > 0:
                loss_mse += self.criterion_mse(y_pred[layer_idx], y_true[layer_idx])

        loss_xent = loss_xent * self.xent_factor / len(y_pred)
        loss_mse = loss_mse * self.mse_factor / len(y_pred)
        loss = loss_xent + loss_mse

        outputs = {'loss_xent': loss_xent.item() if self.xent_factor > 0 else 0,
                   'loss_mse': loss_mse.item() if self.mse_factor > 0 else 0, 
                   'mse_factor': self.mse_factor, 
                   'xent_factor': self.xent_factor,
                   'layer_idx': self.layer_idx}
        return loss, outputs
