"""
Hedgehog attention combining sliding window and linear attentions

For each layer: 
- We first compute (softmax) attention over sliding windows
- We then compute standard linear attention to "fill in" the earlier parts
- We combine to model the entire sequence
"""
from typing import List, Tuple, Optional, Dict, Any
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache  # Transformers v4.36

# Causal linear attention dot product CUDA kernel from fast-transformers
from csrc import causal_dot_product

from src.model.rotary import apply_rotary_pos_emb
from .linear_attention import (
    LolcatsLinearAttention, LinearAttentionState, softmax_attention, repeat_kv
)

# ----------------------
# Sliding window helpers
# ----------------------

def get_causal_mask(x: torch.Tensor):
    """
    Assume x is shape (..., m, n)
    Return mask of shape (..., m, n) where 1 is include, 0 is mask
    """
    m, n = x.shape[-2:]
    return torch.ones((1, 1, m, n), device = x.device, dtype = int).tril(n - m)


def get_under_window_mask(x: torch.Tensor, window_size: int):
    """Return mask for under window terms"""
    m, n = x.shape[-2:]
    return torch.ones((1, 1, m, n), device=x.device, dtype=int).tril(-window_size)


def get_sliding_window_mask(x: torch.Tensor, window_size: int):
    """Return sliding window mask"""
    mask = get_causal_mask(x)
    return (mask - get_under_window_mask(x, window_size)).to(dtype=mask.dtype)


def hybrid_window_attention_quadratic(q: torch.Tensor, k: torch.Tensor,
                                      f_q: torch.Tensor, f_k: torch.Tensor,
                                      v: torch.Tensor,  window_size: int,
                                      eps: float = 1e-12,):
    """Comput hybrid window attention with quadratic complexity"""
    # 1. Sliding window (softmax attention)
    a_sm = torch.einsum('bhmd,bhnd->bhmn', q, k) * (k.shape[-1] ** -0.5)

    mask_causal = get_causal_mask(a_sm)
    mask_linear = get_under_window_mask(a_sm, window_size)
    mask_window = mask_causal - mask_linear

    a_sm = a_sm.masked_fill(~mask_window.bool(), -torch.finfo(a_sm.dtype).max)
    a_sm = torch.softmax(a_sm, dim=-1)

    # 2. Under window (linear attention)
    a_ln = torch.einsum('bhmd,bhnd->bhmn', f_q, f_k)

    # First compute all causal terms (to normalize with)
    a_causal = a_ln.masked_fill(~mask_causal.bool(), 0)
    sum_a_ca = a_causal.sum(dim=-1, keepdim=True)
    sum_a_ca[sum_a_ca == 0] += eps  # stability

    # Then compute actual linear attn terms
    a_ln = a_ln.masked_fill(~mask_linear.bool(), 0)
    sum_a_ln = a_ln.sum(dim=-1, keepdim=True)
    sum_a_ln[sum_a_ln == 0] += eps  # stability

    a_ln = a_ln / sum_a_ca  # linear attention weights
    ratio_sm = 1 - (sum_a_ln / sum_a_ca)  # ratio allocated to softmax terms

    # 3. Combine
    a = a_ln + ratio_sm * a_sm
    y = torch.einsum('bhmn,bhnd->bhmd', a, v)
    return y, a


def under_window_dot_prod(f_q: torch.Tensor, f_k: torch.Tensor, v: torch.Tensor,
                          window_size: int, eps: float=1e-12):
    """Compute hybrid window attention dot product with linear complexity in q_len"""
    dtype = f_q.dtype
    w   = window_size
    f_k = F.pad(f_k, (0, 0, w, 0), value=0)[:, :, :-w, :]
    v   = F.pad(v, (0, 0, w, 0), value=0)[:, :, :-w, :]
    qkv = causal_dot_product(f_q.contiguous().to(dtype=torch.float32), 
                             f_k.contiguous().to(dtype=torch.float32), 
                             v.contiguous().to(dtype=torch.float32)).to(dtype=dtype)
    sum_f_k = f_k.float().cumsum(dim=2).to(dtype=dtype)
    sum_qk = torch.einsum("bhld,bhld->bhl", f_q, sum_f_k)[..., None]
    sum_qk[sum_qk == 0] += eps
    
    return qkv, sum_qk


def sliding_window_softmax_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                     window_size: int, mask_value: float=-1e8):
    """
    Compute sliding window softmax attention without materializing 
    O(seq_len^2) attention weights
    """        
    d = q.shape[-1]
    # Compute windows for keys
    window_kwargs = {'dimension': 2, 'size': window_size, 'step': 1}
    k = F.pad(k, (0, 0, window_size - 1, 0), value=0).unfold(**window_kwargs)
    v = F.pad(v, (0, 0, window_size - 1, 0), value=0).unfold(**window_kwargs)

    # Compute windowed_softmax(qk); causal in its construction
    qk = torch.einsum('bhld,bhldw->bhlw', q, k) * (d ** -0.5)
    qk[qk == 0] = -torch.finfo(q.dtype).max  # heuristic for zeroing out padding above
    return torch.einsum('bhlw,bhldw->bhld', torch.softmax(qk, dim=-1), v)


def hybrid_window_attention(q: torch.Tensor, k: torch.Tensor,
                            f_q: torch.Tensor, f_k: torch.Tensor,
                            v: torch.Tensor, window_size: int,
                            mask_value: float = -1e-8, eps: float = 1e-12,
                            kv_states: Optional[Tuple[torch.Tensor]] = None,
                           ):
    """Compute hybrid sliding window attention with linear complexity"""
    window_kwargs = {'dimension': 2, 'size': window_size, 'step': 1}
    # 1. Sliding window (softmax attention)
    y_sm = sliding_window_softmax_attention(q, k, v, window_size, mask_value)

    # 2. Under window (linear attention)
    sum_f_k = f_k.float().cumsum(dim=2).to(dtype=q.dtype)
    sum_qk_causal = torch.einsum("bhld,bhld->bhl", f_q, sum_f_k)[..., None]
    sum_qk_causal[sum_qk_causal == 0] += eps
    qkv_ln, sum_qk_ln = under_window_dot_prod(f_q, f_k, v, window_size)

    # 3. Combine
    y_ln = qkv_ln / sum_qk_causal
    ratio_sm = 1 - (sum_qk_ln / sum_qk_causal)  # ratio allocated to softmax terms
    return y_ln + ratio_sm * y_sm


class LolcatsSlidingWindowAttention(LolcatsLinearAttention):
    """
    LoLCATs attention combining sliding window and linear attention
    """
    def __init__(self, window_size: int = 64, **kwargs):
        self.window_size = window_size
        self.window_kwargs = {'dimension': 2, 'size': window_size, 'step': 1}
        super().__init__(**kwargs)
        self.attention_type = kwargs['attention_type']  #  'hedgehog_llama_window_tk'

    def base_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor = None, 
                       causal: bool = True, **kwargs: any):
        """
        Standard softmax attention
        """
        a = torch.einsum('bhmd,bhnd->bhmn', q, k) * (k.shape[-1] ** -0.5)
        y = None
        if causal:
            m, n = a.shape[-2:]
            causal_mask = torch.ones((m, n), device = a.device, dtype = torch.bool).triu(n - m + 1)
            a = a.masked_fill(causal_mask, -torch.finfo(a.dtype).max)
        a = torch.softmax(a, dim=-1)
        if v is not None:
            y = torch.einsum('bhmn,bhnd->bhmd', a, v)
        return y, a, None

    def process_qkv(self,
                    hidden_states: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None,
                    position_ids: Optional[torch.LongTensor] = None,
                    past_key_value: Optional[Tuple[int, torch.Tensor, torch.Tensor]] = None,):  # "legacy" cache approach
        """
        Compute queries, keys, and values
        """
        b, l, _ = hidden_states.size()
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        kv_seq_len = k.shape[-2]

        # Shape is (batch_size, seq_len, num_heads, head_dim)
        q = q.view(b, l, *self.q_shape).transpose(1, 2)
        k = k.view(b, l, *self.k_shape).transpose(1, 2)
        v = v.view(b, l, *self.v_shape).transpose(1, 2)

        if past_key_value is not None:  #  and k.shape[2] > q.shape[2]:  # e.g., when generating
            past_key_value.window_size = self.window_size
            # print(f'{self.layer_idx} usable length', past_key_value.get_usable_length(kv_seq_len, self.layer_idx))
            past_key_value.window_size = self.window_size
            if isinstance(past_key_value, Cache):  # In Transformers v4.36+ this is a DynamicCache object
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            else:
                kv_seq_len += past_key_value[0].shape[-2]

        # Apply rotary embeddings and repeat for GQA
        if position_ids is not None and kv_seq_len <= position_ids[0, -1]:
            kv_seq_len = position_ids[0, -1] + 1  # hack for adjusting position ids
        try: # As in Transformers v4.36
            cos, sin = self.rotary_emb(k, seq_len=kv_seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        except TypeError:  # As in Transformers v4.39+
            cos, sin = self.rotary_emb(v, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        return q, k, v, kv_seq_len

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Tuple[int, torch.Tensor, torch.Tensor]] = None,  # "legacy" cache approach
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
        q, k, v, kv_seq_len = self.process_qkv(hidden_states, attention_mask, 
                                               position_ids, past_key_value)
        f_q, f_k = self.feature_map_q(q), self.feature_map_k(k)  # Have to do after repeat for grouped-query attn if we use same fmap

        if self.train_attention:
            # 1. Compute "ground-truth" attention output and weights
            with torch.no_grad():
                _y_true, a_true = self.base_attention(q, k, v)[:2]
                y_true = _y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
                y_true = self.o_proj(y_true)

            # 2. Compute "predicted" attention outputs
            # compute attn weights under sliding window
            y_pred, a_pred = hybrid_window_attention_quadratic(q, k, f_q, f_k, v, window_size=self.window_size)
            attn_weights = ((a_pred, a_true), (y_pred, _y_true))
        else:
            # During finetuning and inference
            if attention_mask is not None and f_q.shape[2] > 1:
                if len(attention_mask.shape) == 4:
                    lin_attn_mask = (attention_mask == 0)[:, :1, -1, :][..., None]  # b, 1, k_len, 1
                else:
                    lin_attn_mask = attention_mask[:, None, :, None]  # b, 1, k_len, 1
                f_k = f_k.masked_fill(~lin_attn_mask, self.mask_value)

            if past_key_value is not None:
                past_key_value.window_size = self.window_size
                if f_q.shape[2] == 1 and kv_seq_len > 1 and past_key_value is not None:  # indicates we're generating
                    # print(f'Recurrent view | layer {self.layer_idx} | type(past_key_value): {type(past_key_value)}')
                    assert use_cache is True

                    # Linear attention bit:
                    # 1. Update K cache by first evicting first token 
                    #    (it's outside the sliding window now), and computing feature_map(k)
                    f_k_from_cache = past_key_value.k_cache[self.layer_idx][:, :, :1, :]
                    # MZ 6/3: handle short inputs; zero-out padding when initial k.shape[2] < self.window_size
                    if f_k_from_cache.sum() == 0:  # heuristic for zeroing out padding in cache
                        f_k_from_cache = torch.zeros(f_q.shape, dtype=f_q.dtype, device=f_q.device)
                    else:                           
                        f_k_from_cache = self.feature_map_k(f_k_from_cache)

                    # 2. Then compute feature_map(k) v^T and add to kv_state
                    #    (past_key_value.update takes care of this, gets the proper stored v)
                    #    v in the arg below is handled separately
                    kv_states = past_key_value.update(k, f_k_from_cache, f_k, v, self.layer_idx)
                    kv_state, k_state, k_state_causal = kv_states

                    # 3. Finally compute linear attentions over terms before the window
                    qkv_ln = torch.einsum('bhlf,bhfd->bhld', f_q, kv_state)
                    sum_qk_ln = torch.einsum('bhlf,bhlf->bhl', f_q, k_state)[..., None]
                    sum_qk_causal = torch.einsum('bhlf,bhlf->bhl', f_q, k_state_causal)[..., None]

                    y_ln = qkv_ln / sum_qk_causal
                    ratio_sm = 1 - (sum_qk_ln / sum_qk_causal)

                    # Sliding window attention bit
                    # -> Compute attention over K and V fixed window cache
                    a_sm = torch.einsum('bhmd,bhnd->bhmn', q, past_key_value.k_cache[self.layer_idx])
                    a_sm[a_sm == 0] = -torch.finfo(q.dtype).max  # heuristic for zeroing out padding in cache
                    a_sm = torch.softmax(a_sm * q.shape[-1] ** -0.5, dim=-1)
                    try:
                        y_sm = torch.einsum('bhmn,bhnd->bhmd', a_sm, past_key_value.v_cache[self.layer_idx])
                    except:
                        breakpoint()

                    # Combine
                    y_true = y_ln + ratio_sm * y_sm
                else:
                    past_key_value.init_cache_and_states(k, f_k, v, self.layer_idx)
                    y_true = hybrid_window_attention(q, k, f_q, f_k, v, window_size=self.window_size)                   
            else:
                y_true = hybrid_window_attention(q, k, f_q, f_k, v, window_size=self.window_size)

            # Concatenate heads and apply output projection
            y_true = y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
            y_true = self.o_proj(y_true)
            attn_weights = None

        return y_true, attn_weights, past_key_value


class LinearAttentionSlidingWindowCache(LinearAttentionState):
    """
    Class for `past_key_values`
    -> Alternative to KV cache; here we only maintain a "KV state" and "K state"
    -> Modified from transformers.cache_utils.DynamicCache (v4.36)
    """
    def __init__(self, window_size: int = 64) -> None:
        self._seen_tokens = 0  # Note in Transformer versions >4.36 this all `seen_tokens*` should be `_seen_tokens*`
        # track phi(k)v^T and phi(k) until softmax terms
        self.kv_states: List[torch.Tensor] = []
        self.k_states:  List[torch.Tensor] = []
        # track all phi(k) until causal end
        self.k_states_causal: List[torch.Tensor] = []

        self.k_cache: List[torch.Tensor] = []
        self.v_cache: List[torch.Tensor] = []
        self._seen_tokens_by_layer: List[int] = []
        self.window_size = window_size

    def init_cache_and_states(self,
                              keys: torch.Tensor,
                              fmap_keys: torch.Tensor,
                              values: torch.Tensor,
                              layer_idx: Optional[int] = None):
        """
        Initialize KV cache and states
        """
        if layer_idx == 0:
            self._seen_tokens += keys.shape[-2]

        dtype = keys.dtype

        # MZ 6/3: handle short inputs; pad if k.shape[2] < self.window_size
        if keys.shape[-2] < self.window_size:
            keys = F.pad(keys, (0, 0, self.window_size - keys.shape[-2], 0), value=0)
        k = F.pad(fmap_keys, (0, 0, self.window_size, 0), value=0) # [:, :, :-w, :]
        v = F.pad(values, (0, 0, self.window_size, 0), value=0) # [:, :, :-w, :]

        # k_cache keeps track of k; k_state keeps track of phi(k)
        k_cache, k_state = keys[:, :, -self.window_size:, :], k[:, :, :-self.window_size, :]
        v_cache, v_state = v[:, :, -self.window_size:, :], v[:, :, :-self.window_size, :]
        
        # Update the cache
        self.k_cache.append(k_cache)
        self.v_cache.append(v_cache)

        # kv_state = torch.einsum('bhlf,bhld->bhfd', k_state, v_state)  # b, h, f, d
        # k_state  = k_state.sum(dim=-2, keepdim=True)  # b, h, 1, f; note the 1
        kv_state = torch.einsum('bhlf,bhld->bhfd', k_state.float(), v_state.float()).to(dtype)  # b, h, f, d
        k_state  = k_state.float().sum(dim=-2, keepdim=True).to(dtype)  # b, h, 1, f; note the 1
        self.kv_states.append(kv_state)
        self.k_states.append(k_state)
        
        # Keep track of all qk_sum
        # self.k_states_causal.append(fmap_keys.sum(dim=-2, keepdim=True))                      
        self.k_states_causal.append(fmap_keys.float().sum(dim=-2, keepdim=True).to(dtype))                      
        self._seen_tokens_by_layer[layer_idx] = keys.shape[-2]        

    def update(self,
               keys: torch.Tensor,
               fmap_key_from_cache: torch.Tensor,
               fmap_keys: torch.Tensor,
               values: torch.Tensor,
               layer_idx: Optional[int] = None,
               cache_kwargs: Optional[any] = None,
               *args, **kwargs: any,
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV cache and states during generation
        """
        assert keys.shape[-2] == 1
        dtype = keys.dtype

        if layer_idx == 0:
            self._seen_tokens += keys.shape[-2]

        # Get first key and value in cache (first added)
        k_state = fmap_key_from_cache  # self.k_cache[layer_idx][:, :, :1, :]
        v_state = self.v_cache[layer_idx][:, :, :1, :]
        # kv_state = torch.einsum('bhlf,bhld->bhfd', k_state, v_state)  # b, h, f, d
        kv_state = torch.einsum('bhlf,bhld->bhfd', k_state.float(), v_state.float()).to(dtype)  # b, h, f, d

        # Update the states and cache
        self.kv_states[layer_idx] += kv_state
        self.k_states[layer_idx]  += k_state
        self.k_states_causal[layer_idx] += fmap_keys

        self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx][:, :, 1:, :], keys], dim=-2)
        self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx][:, :, 1:, :], values], dim=-2)
        self._seen_tokens_by_layer[layer_idx] += keys.shape[-2] 
        return self.kv_states[layer_idx], self.k_states[layer_idx], self.k_states_causal[layer_idx]
