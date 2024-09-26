"""
Custom trainer class for distilling attentions. Can substitute for HuggingFace trainer.
"""
import torch
import torch.nn as nn

from .default_lm import OurTrainer as DefaultTrainer
from src.model.convert_model import traverse_layers


class OurTrainer(DefaultTrainer):
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
        inputs = {k: v.to(model.device) for k, v in data.items() if k != 'labels'}

        with torch.no_grad():
            for layer in traverse_layers(model):
                layer.mse_factor = self.mse_factor
                layer.xent_factor = self.xent_factor

        # hack; we stored losses as attentions
        losses_by_layer = model(**inputs, output_attentions=True, use_cache=False).get('attentions')
        
        loss = 0
        loss_mse = 0
        loss_xent = 0
        n_layers = 0  # Number of layers to distill
        softmax_layers = []
        for layer_idx, (_loss, _loss_mse, _loss_xent) in enumerate(losses_by_layer):
            
            n_layers += 1
            loss += _loss
            loss_mse += _loss_mse
            loss_xent += _loss_xent
        loss_mse /= n_layers
        loss_xent /= n_layers
        loss = loss_mse + loss_xent
        # if self.xent_factor > 0:
        #     breakpoint()
        if 'position_ids' in data:
            outputs = {'loss_xent': loss_xent.item() if self.xent_factor > 0 else 0,
                       'loss_mse': loss_mse.item() if self.mse_factor > 0 else 0,
                       'input_len': data['position_ids'].shape[1],
                       'position_ids': data['position_ids'][0].detach().cpu().numpy(),
                       'mse_factor': self.mse_factor,
                       'xent_factor': self.xent_factor,}
        else:
            outputs = {'loss_xent': loss_xent.item() if self.xent_factor > 0 else 0,
                       'loss_mse': loss_mse.item() if self.mse_factor > 0 else 0, 
                       'mse_factor': self.mse_factor, 
                       'xent_factor': self.xent_factor}
        return loss, outputs
