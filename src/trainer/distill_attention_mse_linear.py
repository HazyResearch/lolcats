"""
Custom trainer class for distilling attentions ("attention transfer") over long sequences with recurrent linear attention view. Can substitute for Hugging Face trainer.
"""
import torch
import torch.nn as nn

from tqdm import tqdm

from src.model.modeling_llama import get_attention_cache
from src.model.convert_model import traverse_layers
from .default_lm import OurTrainer as DefaultTrainer


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
                 **kwargs: any):
        super().__init__(model=model, 
                         metric_for_best_model=metric_for_best_model,
                         **kwargs)
        self.criterion_mse = nn.MSELoss(reduction='mean')
        self.mse_factor = mse_factor
        self.xent_factor = 0
        self.compute_loss_backprop = False  # Whether we backprop in self.compute_loss


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
            # Get hidden states
            true_outputs = model(**inputs, output_attentions=True, 
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

        total_seq_len = 0
        position_ids = torch.arange(input_seq_len).view(1, -1)

        loss_mse = 0
        for layer_idx, layer in enumerate(tqdm(traverse_layers(model), desc='Processing layer', 
                                               leave=False)):
            attn_input, attn_output = true_attn_io[layer_idx]
            attn_preds = layer.self_attn(attn_input.to(model.device),
                                         attention_mask=None,
                                         position_ids=position_ids.to(model.device),
                                         past_key_value=past_key_values)[1]
            if self.mse_factor > 0:  # MSE on layer outputs
                loss_mse += self.criterion_mse(attn_preds, attn_output.to(model.device))
            del attn_input; del attn_output
        loss_mse = loss_mse / (layer_idx + 1) * self.mse_factor
        loss = loss_mse
        torch.cuda.empty_cache()

        if 'position_ids' in data:
            outputs = {'loss_mse': loss_mse.item(),
                       'loss_xent': 0,
                       'mse_factor': self.mse_factor, 
                       'xent_factor': self.xent_factor,
                       'input_len': data['position_ids'].shape[1],
                       'position_ids': data['position_ids'][0],}
        else:
            outputs = {'loss_mse': loss_mse.item(),
                       'loss_xent': 0,
                       'mse_factor': self.mse_factor,
                       'xent_factor': self.xent_factor,}
        return loss, outputs