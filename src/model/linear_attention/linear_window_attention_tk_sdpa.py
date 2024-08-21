"""
Use HF SDPA to compute ground-truth attention outputs
"""
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

from transformers.cache_utils import Cache
try:
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
except ModuleNotFoundError:
    _flash_attention_forward = None  # Transformers v4.36

from src.model.rotary import apply_rotary_pos_emb
from .linear_window_attention_tk import LolcatsTKWindowAttention


class LolcatsTKWindowAttentionSDPA(LolcatsTKWindowAttention):
    """
    Lolcats attention combining sliding window and linear attention
    """
    def __init__(self, remove_base_attn=False, **kwargs):
        # keep self.base_attn for SDPA inference
        super().__init__(remove_base_attn=False, **kwargs)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Cache] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                **kwargs,
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with the option to compute attention weights multiple ways
        if self.train_attention is True
        -> Consistent with HuggingFace Transformers for easy use with their pretrained models
        """
        b, l, _ = hidden_states.size()
        if self.train_attention:
            with torch.no_grad():
                _y_true = self.base_attn(hidden_states=hidden_states,
                                         attention_mask=None,
                                         position_ids=position_ids,
                                         past_key_value=None,
                                         output_attentions=False,
                                         # output_hidden_states=False,
                                         use_cache=False)[0]
                # _y_true.shape is (batch_size, num_heads,seq_len, head_dim)
                y_true = _y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
                y_true = self.o_proj(y_true)
        else:
            y_true = None

        q, k, v, kv_seq_len = self.process_qkv(hidden_states, attention_mask,
                                               position_ids, past_key_value)
        f_q, f_k = self.feature_map_q(q), self.feature_map_k(k)

        # attention_mask = None  # For now this is always True
        if past_key_value is None:  # Regular training
            window_factors = F.sigmoid(self.window_factors)
            linear_factors = 1 - window_factors if self.affine_attention_factors else 1
            y_pred, a_pred = self.quadratic_attention(q, k, f_q, f_k, v, 
                                                      window_factors, linear_factors,
                                                      window_size=self.window_size,)
        else:
            past_key_value.window_size = self.decode_window_size
            if f_q.shape[2] == 1 and kv_seq_len > 1 and not self.training:  # Generating
                assert use_cache is True
                _kv = past_key_value.update_for_decoding(k, v, self.layer_idx,
                                                         self.feature_map_k,
                                                         dtype=q.dtype)
                k_cache, v_cache, kv_state, k_state = _kv

                # Sliding window + linear attention decode
                window_factors = F.sigmoid(self.window_factors)
                linear_factors = 1 - window_factors if self.affine_attention_factors else 1

                a_sm = torch.einsum('bhmd,bhnd->bhmn', q.float(), k_cache.float()) * (k.shape[-1] ** -0.5)
                # a_sm = torch.softmax(a_sm, dim=-1)
                a_sm_max = torch.amax(a_sm, dim=-1, keepdim=True)
                a_sm   = window_factors * torch.exp(a_sm - a_sm_max)
                sum_sm = a_sm.sum(dim=-1, keepdim=True)

                y_pred = (torch.einsum('bhmn,bhnd->bhmd', a_sm, v_cache.float())
                          + linear_factors * torch.einsum('bhld,bhdf->bhlf', f_q.float(), kv_state.float()))
                sum_ln = linear_factors * torch.einsum('bhld,bhnd->bhl', f_q.float(), k_state.float())[..., None]
                y_pred = (y_pred / (sum_sm + sum_ln)).to(q.dtype) 

            else:
                window_factors = F.sigmoid(self.window_factors)
                linear_factors = 1 - window_factors if self.affine_attention_factors else 1
                _y_pred, a_pred = self.quadratic_attention(q, k, f_q, f_k, v,
                                                           window_factors, linear_factors,
                                                           window_size=self.window_size,
                                                           kv_state=None,
                                                           k_state=None,)

        # Concatenate heads and apply output projection
        y_pred = self.o_proj(_y_pred.transpose(1, 2).contiguous().view(b, l, self.hidden_size))
        if self.train_attention:
            attn_weights = (None, (_y_pred, _y_true))  # flash_attn outputs are shape (b, l, h, d)
        else:
            attn_weights = None
        return y_pred, attn_weights, past_key_value
