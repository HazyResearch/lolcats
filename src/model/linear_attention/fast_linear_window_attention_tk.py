"""
Subquadratic attention combining sliding window and linear attentions
- Using the TK "terracing" arrangement

For each layer: 
- We first compute (softmax) attention over sliding windows
- We then compute standard linear attention to "fill in" the earlier parts
- We combine to model the entire sequence
"""
from typing import List, Tuple, Optional, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache

from .linear_attention import (
    LolcatsLinearAttention, LinearAttentionState
)

from flash_attn.utils.distributed import get_dim_for_local_rank
from flash_attn import (
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
    flash_attn_func,
)
import inspect
_flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
assert _flash_supports_window_size, "flash_attn_func does not support window_size"

try:
    from fla.ops.based import fused_chunk_based, parallel_based
    from fla.ops.based.naive import naive_parallel_based
    print(f"Successfully imported the FLA triton kernels! ")
except:
    print(f"Could not import the FLA triton kernels... ")
try:
    from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
    from fla.ops.linear_attn import (chunk_linear_attn, fused_chunk_linear_attn,
                                 fused_recurrent_linear_attn)
except:
    print(f"Could not import the FLA triton kernels... ")
from fla.modules import FusedRMSNormSwishGate, RMSNorm
from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.modules.activations import ACT2FN


def hybrid_attention(
    q: torch.Tensor, k: torch.Tensor, 
    f_q: torch.Tensor, f_k: torch.Tensor,
    v: torch.Tensor,
    window_factor: torch.Tensor,
    linear_factor: torch.Tensor,
    window_size: int,
    kv_state: torch.Tensor = None,
    k_state: torch.Tensor = None,
    eps: float = 1e-12,
    **kwargs
):
    # 1. SWA
    v_slide = v.transpose(1,2)
    o_slide = flash_attn_func(
        q, k, v_slide, 0.0, causal=True,
        window_size=(window_size, 0), 
    ).transpose(1,2)

    # 2. Under window (linear attention)
    o, recurrent_state = fused_chunk_linear_attn(
        f_q.float(), f_k.float(), v.float(),  
        normalize=False
    )

    # 3. Combine
    y = linear_factor * o.to(f_q.dtype) + window_factor * o_slide
    return y


class FasterLolcatsTKWindowAttention(LolcatsLinearAttention):
    """
    Lolcats attention combining sliding window and linear attention
    """
    def __init__(
        self, 
        window_size: int = 64, 
        decode_window_size: int = None,
        affine_attention_factors: bool = False,
        init_window_factor: float = 0,
        state_grad_enabled: bool = False,
        **kwargs
    ):
        self.window_size = window_size
        self.decode_window_size = window_size
        self.window_kwargs = {'dimension': 2, 'size': window_size, 'step': 1}
        super().__init__(**kwargs)
        self.quadratic_attention = hybrid_attention

        # Learnable factor for combining attentions
        self.affine_attention_factors = affine_attention_factors
        device, dtype = self.q_proj.weight.device, self.q_proj.weight.dtype
        self.window_factors = nn.Parameter(
            init_window_factor * torch.ones(1, self.num_heads, 1, 1, device=device, dtype=dtype))
        
        # Whether we use original flash attention 2 inference (use during attention transfer)
        self.base_inference = False
        self.state_grad_enabled = state_grad_enabled
        
    def forward(
        self,
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
        q, k, v, kv_seq_len = self.process_qkv(hidden_states, attention_mask, position_ids, past_key_value)
        f_q, f_k = self.feature_map_q(q), self.feature_map_k(k)

        if self.train_attention:
            # 1. Compute "ground-truth" attention output and weights
            with torch.no_grad():
                q_true = q.transpose(1,2)
                k_true = k.transpose(1,2)
                v_true = v.transpose(1,2)
                with torch.no_grad():
                    _y_true = flash_attn_func(
                        q_true, k_true, v_true,
                        0.0, 
                        causal=True,
                    ).transpose(1,2)
                    y_true = _y_true.reshape(b, l, -1).contiguous()
                    y_true = self.o_proj(y_true)

            # 2. Compute "predicted" attention outputs; compute attn under sliding window
            window_factors = F.sigmoid(self.window_factors)
            linear_factors = 1 - window_factors if self.affine_attention_factors else 1
            y_pred = self.quadratic_attention(
                q_true, k_true, f_q, f_k, v,
                window_factors, linear_factors,
                window_size=self.window_size
            )
            attn_weights = ((None, None), (y_pred, _y_true))        
        else:
            attn_weights = None
            # attention_mask = None  # For now this is always True
            if past_key_value is None:  # Regular training
                window_factors = F.sigmoid(self.window_factors)
                linear_factors = 1 - window_factors if self.affine_attention_factors else 1
                y_true, a_pred = self.quadratic_attention(q, k, f_q, f_k, v,
                                                          window_factors, linear_factors,
                                                          window_size=self.window_size)
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

                    # Softmax attention terms
                    a_sm = torch.einsum('bhmd,bhnd->bhmn', q.float(), k_cache.float()) * (k.shape[-1] ** -0.5)
                    a_sm_max = torch.amax(a_sm, dim=-1, keepdim=True)
                    a_sm   = torch.exp(a_sm - a_sm_max)
                    sum_sm = a_sm.sum(dim=-1, keepdim=True)
                    y_slide = torch.einsum('bhmn,bhnd->bhmd', a_sm/sum_sm, v_cache.float()) # CORRECT?

                    y_lin = linear_factors * torch.einsum('bhld,bhdf->bhlf', f_q.float(), kv_state.float())
                    
                    # Combine with linear attention terms
                    y = linear_factor * y_lin.to(f_q.dtype) + window_factor * y_slide

                else:  # Stateful training
                    try:
                        kv_state = past_key_value.kv_states[self.layer_idx]
                        k_state  = past_key_value.k_states[self.layer_idx]
                    except IndexError:
                        kv_state, k_state = None, None
                    window_factors = F.sigmoid(self.window_factors)
                    linear_factors = 1 - window_factors if self.affine_attention_factors else 1
                    y_true, _ = self.quadratic_attention(q, k, f_q, f_k, v,
                                                         window_factors, linear_factors,
                                                         window_size=self.window_size,
                                                         kv_state=kv_state,
                                                         k_state=k_state)
                    # Save and update KV cache and states
                    past_key_value.update(k, v, self.layer_idx,
                                          fmap_key_states=f_k,
                                          accumulate_in_fp32=True)
            # Concatenate heads and apply output projection
            y_true = y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
            y_true = self.o_proj(y_true)
        return y_true, attn_weights, past_key_value


class LinearAttentionTKWindowCache(LinearAttentionState):
    """
    Class for `past_key_values`
    -> Alternative to KV cache; here we only maintain a "KV state" and "K state"
    -> Modified from transformers.cache_utils.DynamicCache (v4.36)
    """
    def __init__(self, window_size: int = 64) -> None:
        super().__init__()
        self._seen_tokens = 0  # should be `self.seen_tokens` in Transformers v4.36
        self._seen_tokens_by_layer: List[int] = []
        self.kv_states: List[torch.Tensor] = []
        self.k_states:  List[torch.Tensor] = []

        # Account for sliding windows
        self.decode_kv_states: List[torch.Tensor] = []
        self.decode_k_states: List[torch.Tensor] = []
        self.k_cache: List[torch.Tensor] = []
        self.v_cache: List[torch.Tensor] = []
        self.window_size = window_size

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, 
               layer_idx: Optional[int] = None, cache_kwargs: Optional[any] = None,
               accumulate_in_fp32: bool = False, 
               fmap_key_states: torch.Tensor = None,  # should not be None
               grad_enabled: bool = False,
               **kwargs: any,
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV, K states; and KV cache during training
        - For decoding, use `self.decode_kv_states` to keep track of KV states 
          up to sliding window terms
        - For (chunked) training, use `self.kv_states` to keep track of KV states
          up to end of sequence
        - Likewise for `self.decode_k_states` and `self.k_states`
        """
        with torch.set_grad_enabled(grad_enabled):
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]

            dtype = key_states.dtype
            if accumulate_in_fp32:
                # key_states = key_states.float()
                fmap_key_states = fmap_key_states.float()
                value_states = value_states.float()

            # Decoding KV state (KV terms up to last window_size)
            decode_kv_state = torch.einsum(
                'bhlf,bhld->bhfd', 
                fmap_key_states[:, :, :-self.window_size],
                value_states[:, :, :-self.window_size]
            )
            # KV state
            kv_state = decode_kv_state + torch.einsum(
                'bhlf,bhld->bhfd', 
                fmap_key_states[:, :, -self.window_size:],
                value_states[:, :, -self.window_size:]
            )
            # shape is b, h, 1, f; note the 1
            decode_k_state = fmap_key_states[:, :, :-self.window_size].sum(dim=-2, keepdim=True)
            k_state = (decode_k_state +
                       fmap_key_states[:, :, -self.window_size:].sum(dim=-2, keepdim=True))

            # Update the cache
            if len(self.k_states) <= layer_idx:  # Initializing kv and k states
                self.kv_states.append(kv_state.to(dtype))
                self.k_states.append(k_state.to(dtype))

                self.decode_kv_states.append(decode_kv_state.to(dtype))
                self.decode_k_states.append(decode_k_state.to(dtype))

                self.k_cache.append(key_states[:, :, -self.window_size:, :])
                self.v_cache.append(value_states[:, :, -self.window_size:, :].to(dtype))
                # self._seen_tokens_by_layer[layer_idx].append(key_states.shape[-2])
            else:
                # Update kv and k states recurrently
                kv_state = (self.kv_states[layer_idx].to(kv_state.dtype) + kv_state).to(dtype)
                k_state  = (self.k_states[layer_idx].to(kv_state.dtype) + k_state).to(dtype)
                self.kv_states[layer_idx] = kv_state
                self.k_states[layer_idx]  = k_state

                decode_kv_state = (self.decode_kv_states[layer_idx].to(kv_state.dtype) 
                                   + decode_kv_state).to(dtype)
                decode_k_state  = (self.decode_k_states[layer_idx].to(kv_state.dtype) 
                                   + decode_k_state).to(dtype)
                self.decode_kv_states[layer_idx] = decode_kv_state
                self.decode_k_states[layer_idx]  = decode_k_state

                self.k_cache[layer_idx] = key_states[:, :, -self.window_size:, :]
                self.v_cache[layer_idx] = value_states[:, :, -self.window_size:, :]
            self._seen_tokens_by_layer[layer_idx] += key_states.shape[-2]

        return self.kv_states[layer_idx], self.k_states[layer_idx]

    def update_for_decoding(self, keys: torch.Tensor, values: torch.Tensor, 
                            layer_idx: int, feature_map_k: Callable, dtype: torch.dtype):
        """
        Update the decoding KV and K states, and KV cache, during decodeing
        """
        with torch.no_grad():
            k_cache = self.k_cache[layer_idx]
            v_cache = self.v_cache[layer_idx]
            k_state = feature_map_k(k_cache[:, :, :1, :])
            v_state = v_cache[:, :, :1, :]
            kv_state = torch.einsum('bhlf,bhld->bhfd', k_state.float(), v_state.float()).to(dtype) # b, h, f, d
            self.decode_kv_states[layer_idx] += kv_state
            self.decode_k_states[layer_idx] += k_state
            
            self.k_cache[layer_idx] = torch.cat([k_cache[:, :, 1:, :], keys], dim=-2)
            self.v_cache[layer_idx] = torch.cat([v_cache[:, :, 1:, :], values], dim=-2)
            
            if layer_idx == 0:
                self._seen_tokens += keys.shape[-2]
            self._seen_tokens_by_layer[layer_idx] += keys.shape[-2]
            return (self.k_cache[layer_idx], self.v_cache[layer_idx], 
                    self.decode_kv_states[layer_idx], self.decode_k_states[layer_idx])
