"""
Custom trainer class for distilling attentions. Can substitute for HuggingFace trainer.
"""
import torch
import torch.nn as nn

from .default_lm import OurTrainer as DefaultTrainer


class OurTrainer(DefaultTrainer):
    """
    Custom trainer class for distilling attentions. 
    - We compute and store the attention outputs and/or weights for each head and layer,
      for both the "teacher" softmax attentions and "student" learnable subquadratic attentions
    - We then train the student layers to minimize either MSE(outputs) or CrossEntropy(weights)
    """
    def __init__(self,
                 model: nn.Module,  # attention layer
                 teacher_layer: nn.Module,
                 layer_idx: int,
                 metric_for_best_model: str = 'ft/eval/loss',
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
        self.layer_idx = layer_idx
        self.initial_eval = False  # Whether to evaluate before training

        self.teacher_layer = teacher_layer
        self.teacher_layer.eval()
        for p in self.teacher_layer.parameters():
            p.requires_grad = False # freeze teacher

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
            _n = data['inputs_embeds'].shape[-2]  # construct attention mask for ground-truth softmax
            y_true = self.teacher_layer(**inputs, output_attentions=True, output_hidden_states=True, use_cache=False)
            y_true, a_true = y_true.get('hidden_states'), y_true.get('attentions')

        # Student outputs
        y_pred = model(**inputs, output_attentions=True, output_hidden_states=True, use_cache=False)
        y_pred, a_pred = y_pred.get('hidden_states'), y_pred.get('attentions')
    
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