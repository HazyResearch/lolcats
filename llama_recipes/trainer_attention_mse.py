"""
Compute MSE losses using attention outputs
"""
import torch
import torch.nn as nn


class LossComputer():
    """
    Computes the loss for attention distillation
    """
    def __init__(self, mse_factor: int = 1000, xent_factor: int = 1, **kwargs: any) -> None:
        super().__init__()
        self.criterion_mse = nn.MSELoss(reduction='mean')
        self.criterion_xent = nn.CrossEntropyLoss(reduction='mean')
        self.mse_factor = mse_factor
        self.xent_factor = xent_factor

    def compute_loss(self, model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Compute the loss for attention distillation"""
        loss = 0
        loss_mse = 0
        loss_xent = 0
        n_layers = 0  # Number of layers to distill
        outputs = model(**inputs, output_attentions=True, use_cache=False).get('attentions')        
        for _, attns in enumerate(outputs):
            if attns is not None:
                if self.xent_factor > 0:
                    # Cross-entropy loss
                    a_pred, a_true = attns[0]
                    a_pred = a_pred.clamp(min=1e-12).log()  # nn.CrossEntropy assumes unnormalized logits
                    k_len = a_true.shape[-1]  # batch, n_heads, q_len, k_len
                    # Compute mean cross-entropy over all queries
                    a_pred = a_pred.contiguous().view(-1, k_len)
                    a_true = a_true.contiguous().view(-1, k_len)
                    loss_xent += self.criterion_xent(a_pred, a_true)
                    # loss_xent += self.criterion_xent(a_pred.to(model.device), 
                    #                                  a_true.to(model.device))
                if self.mse_factor > 0:
                    loss_mse += self.criterion_mse(*attns[1])
                    # loss_mse += self.criterion_mse(*[a.to(model.device) for a in attns[1]])
                n_layers += 1
                # del attns
                # torch.cuda.empty_cache()
        if n_layers > 0:
            loss_xent = loss_xent * self.xent_factor
            loss_mse = loss_mse * self.mse_factor
        loss = (loss_xent + loss_mse) / n_layers
        return loss