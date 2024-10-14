# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import math
import os
import torch
import time
from collections import OrderedDict
from torch import nn
from torch.nn.parameter import Parameter
from transformers import LlamaConfig

from vllm.attention import Attention, AttentionMetadata, AttentionType
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.distributed import (divide, 
                              # split_tensor_along_last_dim,
                              # tensor_model_parallel_all_gather,
                              # tensor_model_parallel_all_reduce
                              )
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    get_compressed_tensors_cache_scale)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.utils import is_hip



from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.model_executor.models.utils import PPMissingLayer, is_pp_missing_parameter, make_layers
# from .interfaces import SupportsLoRA
# from .utils import PPMissingLayer, is_pp_missing_parameter, make_layers
from vllm.model_executor.models.llama import LlamaForCausalLM, LlamaAttention, LlamaMLP

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear

logger = init_logger(__name__)


### OURS for Linear attention implementation
# from peft import get_peft_model, LoraConfig, TaskType

# PEFT_KWARGS = {
#     'r': 8,
#     'lora_alpha': 16, # 32
#     'lora_dropout': 0.05,
#     'target_modules': ["q_proj", "v_proj", "k_proj",  "o_proj"]
# }

### Hybrid Attention
from vllm.attention import Attention, AttentionMetadata


### VLLM Llama Model
class FeatureMap(nn.Module):
    """
    Learnable MLP in feature map.
    
    Full feature map is like f(xW + b)
    -> This is the `W` and (optional) `b` part
    """
    def __init__(self, 
                 num_heads: int,
                 head_dim: int,
                 feature_dim: int,
                 dtype: torch.dtype,
                 device: torch.device,
                 eps: float = 1e-12,
                 **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.feature_dim = feature_dim
        self.dtype = dtype
        self.device = device
        self.eps = eps
        self.init_weights_()

    def activation(self, x: torch.Tensor):
        return torch.cat([
            torch.softmax(x, dim=-1), torch.softmax(-x, dim=-1)
        ], dim=-1).clamp(min=self.eps)

    def init_weights_(self):
        self.layer = nn.Parameter(torch.zeros(
            (self.num_heads, self.head_dim, self.feature_dim),
            dtype=self.dtype, device=self.device,
        ))

    def forward(self, x: torch.Tensor):
        return self.activation(
            torch.einsum('hdf,bhld->bhlf', self.layer, x.to(self.dtype)))


from dataclasses import dataclass
@dataclass
class LoLCacheParams:
    is_prompt: bool = False
    kv_state: torch.Tensor = torch.Tensor()
    k_state: torch.Tensor = torch.Tensor()
    kv_cache: torch.Tensor = torch.Tensor()

@dataclass
class PageCache:
    kv_cache: torch.Tensor = None
    q_cache: torch.Tensor = None


class LlamaLolcatsAttentionActual(nn.Module):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        num_kv_heads: int,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.window_size = 64

        # SA: inference cache
        self.lolcats_cache = None
        self.layer_idx = layer_idx
        self.tp_rank = get_tensor_model_parallel_rank()


    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        fmap_q: FeatureMap,
        fmap_k: FeatureMap,
        window_factors: torch.Tensor,
        state=None,
        attn_metadata: AttentionMetadata = None
    ) -> torch.Tensor:
        # if self.layer_idx == 0:
        #     print(f"Initially: {query.shape=}, {key.shape=}, {value.shape=}")
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        positions = attn_metadata.seq_start_loc.tolist()
        start, end = positions[0], positions[1]
        
        if self.lolcats_cache is None or end == num_prefill_tokens:
            # reset cache
            self._prepare_lolcats_cache()
            if self.layer_idx == 0 and self.tp_rank == 0:
                print("Resetting cache")
                print(f"-- {num_prefill_tokens=}, {num_decode_tokens=}, {start=}, {end=}")
                # print(self.page_cache.kv_cache)

        # num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
    
        if self.num_kv_heads != self.num_heads:
            key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
            value = value.repeat_interleave(self.num_queries_per_kv, dim=1)

        query = query.movedim(0, query.dim() - 2)
        key = key.movedim(0, key.dim() - 2)
        value = value.movedim(0, value.dim() - 2)

        if query.dim() == 3:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)

        query, key, value = self.get_full_key_value(query, key, value)
        if self.layer_idx == 0 and self.tp_rank == 0:
            print(f"-- after update: {query.shape=}, {key.shape=}, {value.shape=}")

        f_q = fmap_q(query)
        f_k = fmap_k(key)

        window_size = 64
        window_factors = torch.nn.functional.sigmoid(window_factors)
        linear_factors = 1

        seq_len = query.shape[-2]
        if num_decode_tokens >= 1 or seq_len == 1:
            return self.recurrent_attention(
                query, key, f_q, f_k, 
                value, window_factors, 
                linear_factors,
                window_size,
                fmap_q, fmap_k
            )
        else:
            out = self.superlinear_attention(
                query, key, f_q, f_k, 
                value,
                window_factors, linear_factors,
                window_size
            )
            return out
    

    def get_full_key_value(self, query, key, value):
        # add the current key and value to the cache
        if self.page_cache.kv_cache is not None:
            key = torch.cat([self.page_cache.kv_cache[:, 0], key], dim=-2)
            value = torch.cat([self.page_cache.kv_cache[:, 1], value], dim=-2)
            query = torch.cat([self.page_cache.q_cache, query], dim=-2)
        else:
            key = key
            value = value
            query = query

        # update the cache
        self.page_cache.kv_cache = torch.stack([key, value], dim=1)
        self.page_cache.q_cache = query
        return query, key, value


    def _create_mask(self, max_seq_len: int, window_size: int, is_window: bool) -> torch.Tensor:
        l = window_size
        m = math.ceil(max_seq_len / window_size)
        mask = torch.block_diag(*[torch.ones((l, l))] * m)
        mask += torch.roll(mask, -l, -1)
        mask = mask[:max_seq_len, :max_seq_len]
        mask = mask[None, None, ...]  # b, h, q_len, k_len
        mask = torch.tril(mask if is_window else 1 - mask).to(dtype=torch.bool)
        return mask


    def get_masks(self, window_size: int, q_len: int, k_len: int, device: torch.device) -> tuple[torch.Tensor]:
        mask_window = self._create_mask(q_len, window_size, True).to(device)
        mask_linear = self._create_mask(q_len, window_size, False).to(device)
        return mask_window[:, :, :q_len, :k_len], mask_linear[:, :, :q_len, :k_len]


    def _prepare_lolcats_cache(self):
        self.lolcats_cache = LoLCacheParams()
        self.page_cache = PageCache()


    def _init_kv_cache(self, keys, values, f_k):
        dtype = keys.dtype
        
        # decoding KV state (KV terms up to last window_size)
        decode_kv_state = torch.einsum('bhlf,bhld->bhfd', 
            f_k[:, :, :-self.window_size],
            values[:, :, :-self.window_size]
        )

        # shape is b, h, 1, f; note the 1
        decode_k_state = f_k[:, :, :-self.window_size].sum(dim=-2,keepdim=True)
        self.lolcats_cache.kv_state = decode_kv_state
        self.lolcats_cache.k_state = decode_k_state

        # update the cache
        kv_cache = torch.stack([
            keys[:, :, -self.window_size:, :].float(),
            values[:, :, -self.window_size:, :].float()
        ], dim=1)
        self.lolcats_cache.kv_cache = kv_cache


    def superlinear_attention(
        self, q: torch.Tensor, k: torch.Tensor, 
        f_q: torch.Tensor, f_k: torch.Tensor,
        v: torch.Tensor,
        window_factor: torch.Tensor, linear_factor: torch.Tensor,
        window_size: int, 
        kv_state: torch.Tensor = None, k_state: torch.Tensor = None,
        eps: float = 1e-12,
        mask_value: float=-1e8
    ):
        """
        Hybrid attention combining sliding window and linear attentions
        """

        mask_window, mask_linear = self.get_masks(
            window_size, q.shape[-2], k.shape[-2], q.device)
        
        # 1. Sliding window (softmax attention)
        a_sm = torch.einsum(
            'bhmd,bhnd->bhmn', q.float(), k.float()) * (k.shape[-1] ** -0.5)
        a_sm = a_sm.masked_fill(~mask_window.bool(), mask_value)
        # torch.softmax(a_sm, dim=-1), but we account for the max when combining
        a_sm_max = torch.amax(a_sm, dim=-1, keepdim=True)
        a_sm   = window_factor * torch.exp(a_sm - a_sm_max)
        sum_sm = a_sm.sum(dim=-1, keepdim=True)

        # 2. Under window (linear attention)
        a_ln = torch.einsum('bhmd,bhnd->bhmn', f_q.float(), f_k.float())
        a_ln = linear_factor * a_ln.masked_fill(~mask_linear.bool(), 0).to(q.dtype)
        sum_ln = a_ln.sum(dim=-1, keepdim=True)

        # 3. Combine
        # Allow outputs to also depend on prior kv_state and k_state
        y = torch.einsum('bhmn,bhnd->bhmd', a_sm + a_ln, v)
        y = (y / (sum_sm + sum_ln))

        self._init_kv_cache(k, v, f_k)
        return y.to(q.dtype) # attention weights only for the last chunk


    def _update_kv_cache(self, keys, values, fmap_k):
        # if self.tp_rank == 0 and self.layer_idx == 0: print("heyo 1 - hello update kv cache")
        # get state from before
        kv_state = self.lolcats_cache.kv_state
        k_state  = self.lolcats_cache.k_state
        kv_cache_swa = self.lolcats_cache.kv_cache
        k_cache = kv_cache_swa[:, 0]
        v_cache = kv_cache_swa[:, 1]

        dtype = kv_state.dtype

        # update the linear attention states
        # since we ignore the diag blocks, just grab last tokens of kv cache
        cur_seq_len = k_cache.shape[-2]
        # if self.tp_rank == 0 and self.layer_idx == 0: print(f"{cur_seq_len=}")
        if cur_seq_len >= self.window_size:
            # if self.tp_rank == 0 and self.layer_idx == 0: print(f"Updating the kv_state and k_state...")
            k_state = fmap_k(k_cache[:, :, :1, :]) 
            v_state = v_cache[:, :, :1, :]
            kv_state = torch.einsum('bhlf,bhld->bhfd', k_state.float(), v_state.float()).to(dtype) # b, h, f, d
            self.lolcats_cache.kv_state += kv_state.to(kv_state.dtype) 
            self.lolcats_cache.k_state += k_state

        # update swa states
        if cur_seq_len < self.window_size:
            # only add to cache
            k_cache = torch.cat([k_cache, keys], dim=-2)
            v_cache = torch.cat([v_cache, values], dim=-2)
        else:
            # remove oldest key and value and append
            k_cache = torch.cat([k_cache[:, :, 1:, :], keys], dim=-2)
            v_cache = torch.cat([v_cache[:, :, 1:, :], values], dim=-2)
        kv_cache_swa = torch.stack([k_cache, v_cache], dim=1)
        self.lolcats_cache.kv_cache = kv_cache_swa

        # if self.tp_rank == 0 and self.layer_idx == 0: print("heyo 2 - bye update kv cache")
        return self.lolcats_cache.kv_state, self.lolcats_cache.k_state, k_cache, v_cache 


    def recurrent_attention(
        self, q: torch.Tensor, k: torch.Tensor, 
        f_q: torch.Tensor, f_k: torch.Tensor,
        v: torch.Tensor,
        window_factor: torch.Tensor,
        linear_factor: torch.Tensor,
        window_size: int,
        fmap_q, fmap_k,
        kv_state: torch.Tensor = None,
        k_state: torch.Tensor = None,
        eps: float = 1e-12, mask_value: float=-1e8
    ):
        dtype = torch.float32
        kv_state, k_state, k_cache, v_cache  = self._update_kv_cache(k, v, fmap_k)
        
        # Softmax attention terms
        a_sm = torch.einsum('bhmd,bhnd->bhmn', q.float(), k_cache.float()) * (k.shape[-1] ** -0.5)
        a_sm_max = torch.amax(a_sm, dim=-1, keepdim=True)
        a_sm   = window_factor * torch.exp(a_sm - a_sm_max)
        sum_sm = a_sm.sum(dim=-1, keepdim=True)
        y_sm = torch.einsum('bhmn,bhnd->bhmd', a_sm.float(), v_cache.float())

        # Combine with linear attention terms
        f_q = fmap_q(q)
        y_ln = linear_factor * torch.einsum('bhlf,bhfd->bhld', f_q.float(), kv_state.float())
        sum_ln = linear_factor * torch.einsum('bhld,bhnd->bhl', f_q.float(), k_state.float())[..., None]

        y = y_sm + y_ln 
        attn_output = (y / (sum_sm + sum_ln)).to(q.dtype)
        return attn_output


class LlamaLolcatsAttention(LlamaAttention):
    def __init__(self, layer_idx, use_base_attn, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.use_base_attn = use_base_attn
        if self.use_base_attn: 
            # coppy the original self.attn into self.base_attn before we override
            # use deepcopy to avoid any shared references
            import copy
            self.base_attn = copy.deepcopy(self.attn)

        self.attn = LlamaLolcatsAttentionActual(self.num_heads,
                                                self.head_dim,
                                                self.num_kv_heads,
                                                layer_idx)
        self.head_size = self.head_dim
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        _device = self.qkv_proj.weight.device
        _dtype = self.qkv_proj.weight.dtype

        _feature_dim = 64
        _feature_map_kwargs = {
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "feature_dim": _feature_dim,
            "dtype": _dtype,
            "device": _device,
        }
        self.feature_dim = _feature_dim
        self.window_size = 64

        tp_rank = get_tensor_model_parallel_rank()
        self.tp_rank = tp_rank
        self.layer_idx = layer_idx

        self.feature_map_q = FeatureMap(**_feature_map_kwargs)
        self.feature_map_k = FeatureMap(**_feature_map_kwargs)
        self.window_factors = nn.Parameter(
            torch.ones(1, self.num_heads, 1, 1, device=_device, dtype=_dtype))


    def load_window_factors(self, loaded_weight: torch.Tensor):
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1:

            num_heads_per_rank = self.num_heads
            start_idx = tp_rank * num_heads_per_rank
            end_idx = start_idx + num_heads_per_rank

            if self.layer_idx == 0 and tp_rank == 0:
                print(loaded_weight)

            if self.layer_idx < 2: 
                print(f"{num_heads_per_rank=}")
                print(f"{tp_rank=}; {loaded_weight.shape=}; {start_idx=}; {end_idx=}")

            sharded_weight = loaded_weight[:, start_idx:end_idx, :, :]
        
        else:

            sharded_weight = loaded_weight

        assert self.window_factors.shape == sharded_weight.shape, \
            f"Shape mismatch: {self.window_factors.shape} vs {sharded_weight.shape}"

        with torch.no_grad():
            self.window_factors.copy_(sharded_weight)

    def load_feature_map_q(self, loaded_weight: torch.Tensor):
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        if tp_size > 1:
        
            num_heads_per_rank = self.feature_map_q.num_heads
            start_idx = tp_rank * num_heads_per_rank
            end_idx = start_idx + num_heads_per_rank

            sharded_weight = loaded_weight[start_idx:end_idx, :, :]

            if sharded_weight.shape[-1] != self.feature_map_q.layer.shape[-1]:
                sharded_weight = sharded_weight[:, :, :self.feature_map_q.layer.shape[-1]]
        
        else:

            sharded_weight = loaded_weight

        assert self.feature_map_q.layer.shape == sharded_weight.shape, \
            f"Shape mismatch: {self.feature_map_q.layer.shape} vs {sharded_weight.shape}"

        with torch.no_grad():
            self.feature_map_q.layer.copy_(sharded_weight)

    def load_feature_map_k(self, loaded_weight: torch.Tensor):
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        if tp_size > 1:
        
            num_heads_per_rank = self.feature_map_k.num_heads
            start_idx = tp_rank * num_heads_per_rank
            end_idx = start_idx + num_heads_per_rank

            sharded_weight = loaded_weight[start_idx:end_idx, :, :]

            if sharded_weight.shape[-1] != self.feature_map_k.layer.shape[-1]:
                sharded_weight = sharded_weight[:, :, :self.feature_map_k.layer.shape[-1]]

        else:

            sharded_weight = loaded_weight

        assert self.feature_map_k.layer.shape == sharded_weight.shape, \
            f"Shape mismatch: {self.feature_map_k.layer.shape} vs {sharded_weight.shape}"

        with torch.no_grad():
            self.feature_map_k.layer.copy_(sharded_weight)
            # self.feature_map_k.layer.normal_(std=1)

    def merge_lora_to_qkv_parallel(self, # param: Parameter, 
                                   loaded_delta: torch.Tensor,
                                   loaded_shard_id: str = 'q',
                                   total_num_heads: int = 32, 
                                   total_num_kv_heads: int = 4,
                                   head_size: int = 128):
        """
        Merge computed delta_AB into QKV parallel weights

        Based off of vLLM linear layer: https://github.com/vllm-project/vllm/blob/bc6e42a9b19364e07da9f279edd81796541d147d/vllm/model_executor/layers/linear.py#L762
        then Rahul, then Claude 3.5 Sonnet

        model.layers.1.self_attn.qkv_proj.weight torch.Size([1280, 8192])
        --> output_dim 0
        model.layers.1.self_attn.o_proj.weight torch.Size([8192, 1024])
        --> output_dim 0

        apply this three times for q, k, and v LoRA deltas to the same layer
        """

        param = self.qkv_proj.weight
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        # num_heads = divide(total_num_heads, tp_size)
        # if tp_size >= total_num_kv_heads:
        #     num_kv_heads = 1
        #     # num_kv_head_replicas = divide(tp_size, total_num_kv_heads)
        # else:
        #     num_kv_heads = divide(total_num_kv_heads, tp_size)
        #     # num_kv_head_replicas = 1
        num_heads = total_num_heads
        num_kv_heads = total_num_kv_heads

        num_original_kv_heads = 8 # all Llama 3.1 models have 8 kv heads in total

        num_kv_head_replicas = tp_size // num_original_kv_heads

        if loaded_shard_id == "q":
            shard_offset = 0
            shard_size = num_heads * head_size
        elif loaded_shard_id == "k":
            shard_offset = num_heads * head_size
            shard_size = num_kv_heads * head_size
        elif loaded_shard_id == "v":
            shard_offset = (num_heads + num_kv_heads) * head_size
            shard_size = num_kv_heads * head_size

        # print(f"{tp_rank=}, {tp_size=}")
        if loaded_shard_id == "q":
            start_idx = tp_rank * shard_size
        else:
            start_idx = (tp_rank // num_kv_head_replicas) * shard_size

        device = param_data.device

        param_data = param_data.narrow(output_dim, shard_offset, shard_size)
        # print(f'{loaded_shard_id=}')
        # print(f'{shard_offset=}, {shard_size=}, {shard_offset+shard_size=}')
        # print(f'{output_dim=}, {start_idx=}, {param_data.shape=}')
        # print('-' * 10)

        # self.qkv_proj.weight.data[shard_offset:shard_offset+shard_size, :] += (
        #     loaded_delta.narrow(output_dim, start_idx, shard_size).to(device)
        # )
        # print(f'Loading {loaded_shard_id} {start_idx}-{start_idx+shard_size} into {param_data.shape}, which is slice({shard_offset}, {shard_offset+shard_size})')
        try:
            param_data.copy_(param_data + loaded_delta.narrow(output_dim, start_idx, shard_size).to(device))
            # print(f"Loaded {loaded_shard_id} into {param_data.shape}")
        except Exception as e:
            print(f"Error: {e}")
            print(f"{loaded_shard_id=}")
            print(f"{output_dim=}")
            print(f"{start_idx=}")
            print(f"{shard_size=}")
            print(f"{param_data.shape=}")
            print(f"{loaded_delta.shape=}")
            print(f"{tp_rank=}")
            print(f"{tp_size=}")

    def merge_lora_to_o_parallel(self, 
                                 loaded_delta: torch.Tensor):
        """
        Merge computed delta_AB into output projection (RowParallel linear)
        """      
        param = self.o_proj.weight
        param_data = param.data
        input_dim = getattr(param, "input_dim", None)
        device = param_data.device

        # print('o_proj {input_dim=}')

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()        
        
        if input_dim is not None:
            shard_size = param_data.shape[input_dim]
            start_idx = tp_rank * shard_size
            loaded_delta = loaded_delta.narrow(input_dim, start_idx, shard_size)

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_delta.shape) == 0:
            loaded_delta = loaded_delta.reshape(1)

        # print('{param_data.shape=} | {loaded_delta.shape=}')
        # assert param_data.shape == loaded_delta.shape
        param_data.copy_(param_data + loaded_delta.to(device))
      
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        problem_idx: int
    ) -> torch.Tensor:
        ndim = hidden_states.dim()
        qkv, _ = self.qkv_proj(hidden_states)
        seq_len = hidden_states.shape[-2]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(
            q, k, v,
            fmap_q=self.feature_map_q,
            fmap_k=self.feature_map_k,
            window_factors=self.window_factors,
            state=None,
            attn_metadata=attn_metadata
        )

        ref_output = None
        expt_tag = '_cria_alpaca_final'
        if self.use_base_attn and self.layer_idx % 9 == 0:
            ref_output = self.base_attn(
                q, k, v,
                attn_metadata=attn_metadata, 
                kv_cache=kv_cache,
            )

            dir_path = f"/data/simran/mmlu_hybrid_outputs_{expt_tag}/"
            if not os.path.exists(dir_path): os.makedirs(dir_path, exist_ok=True)
            fpath = f"{dir_path}/our_attn_output_problem{problem_idx}_rank{self.tp_rank}_layer{self.layer_idx}.pt"
            torch.save(attn_output, fpath)
            fpath = f"{dir_path}/ref_attn_output_problem{problem_idx}_rank{self.tp_rank}_layer{self.layer_idx}.pt"
            torch.save(ref_output, fpath)
            # print(f"Saved!")
            # end save stuff

        # outputs
        full_seq_len = attn_output.shape[-2] # in case we updated the length
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            -1, full_seq_len, self.num_heads * self.head_dim
        )
        output, _ = self.o_proj(attn_output)
        if output.dim() > ndim:
            output = output.squeeze(0)
        output = output[-seq_len:, ...]  # put back the original seq_len

        if self.use_base_attn and self.layer_idx % 9 == 0:
            ref_y, _ = self.o_proj(ref_output)
            dir_path = f"/data/simran/mmlu_hybrid_y_outs_{expt_tag}/"
            if not os.path.exists(dir_path): os.makedirs(dir_path, exist_ok=True)
            fpath = f"{dir_path}/our_y_out_problem{problem_idx}_rank{self.tp_rank}_layer{self.layer_idx}.pt"
            torch.save(output, fpath)
            fpath = f"{dir_path}/ref_y_out_problem{problem_idx}_rank{self.tp_rank}_layer{self.layer_idx}.pt"
            torch.save(ref_y, fpath)

        return output


class LlamaLolcatsForCausalLM(LlamaForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"LOLCATS!!!: Loading model with config: {self.config}")

        tp_rank = get_tensor_model_parallel_rank()
        self.tp_rank = tp_rank

        softmax_attentions = getattr(self.config, 'softmax_attentions', [])
        print(f"{softmax_attentions=}")

        use_base_attn = getattr(self.config, 'use_base_attn', False)

        for i in range(len(self.model.layers)):
            if i in softmax_attentions:
                pass 
            else:
                self.model.layers[i].self_attn = LlamaLolcatsAttention(
                    i,
                    use_base_attn=use_base_attn,
                    config=self.config,
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_attention_heads,
                    num_kv_heads=getattr(self.config, "num_key_value_heads",
                                        self.config.num_attention_heads),
                    rope_theta=self.config.rope_theta,
                    rope_scaling=self.config.rope_scaling,
                )
        print(self.model)


    def get_device(self):
        device = next(self.parameters()).device
        return str(device)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        super().load_weights(weights)

        # r = 8
        # lora_alpha = 16
        # lora_dropout = 0

        # model_size = 8
        # FINETUNE_PATH = '/home/rahul/code/lolcats/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2-m=distill_llama3_1_8b_lk_smd_wtk64_fd64_w01-f=finetune_lora_qkvo_alpaca_clean-s=0-ms=2500-se=0-re=100-lzi=1-bs=1-gas=8-nte=2-ms=2500-se=0-re=100_ft.pt'
        # MLP_PATH = '/home/rahul/code/lolcats/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2-m=distill_llama3_1_8b_lk_smd_wtk64_fd64_w01-f=finetune_lora_qkvo_alpaca_clean-s=0-ms=2500-se=0-re=100-lzi=1_distill.pt'
        # # merge the MLP and FINETUNE weights as adapter weights
        # adapter_weights = torch.load(FINETUNE_PATH, weights_only=True)
        # adapter_weights.update(torch.load(MLP_PATH, weights_only=True))
        # print(adapter_weights.keys())
        # # only keep any weight with 'feature' or 'window' or 'lora' in the key
        # adapter_weights = {k: v for k, v in adapter_weights.items() if 'feature' in k or 'window' in k or 'lora' in k}

        # model_size = 70
        # # PATH = f'/data/rahul/checkpoints/{model_size}b.pt'
        # PATH = f'/home/rahul/code/lolcats/ckpt_lora-dl-d=distill_redpajama_xent1_mse1000_lr1e-2-m=distill_llama3_1_70b_lk_smd_wtk64_fd64_w01-f=finetune_lora_qkvo_redpajama-dcs=512-se=0-re=4-lzi=1-dcs=512-se=0-re=4.pt'

        ########### 405 B ############

        # PATH = '/home/rahul/code/lolcats/ckpts/seqlen768.pt'  # 405B at 768 seqlen

        # 1. Alpaca Cria QV Rank 4 -- with hybridization
        PATH = '/home/rahul/code/lolcats/ckpt_lora-dl-d=distill_llama_405b_xent1_mse1000_lr1e-2-m=distill_llama3_1_405b_lk_smd_wtk64_fd64_w01_h72_80_117_125-f=finetune_layer_mini_xent1_mse1000-s=0-se=0-re=0-ef=finetune_llama_405b_qkvo_e2_h72_80_117_125-ft_lora=0-se=0-re=0-alpaca.pt'

        # 2. Alpaca Cria QV Rank 4 -- pure
        # PATH = '/home/rahul/code/lolcats/ckpt_lora-dl-d=distill_llama_405b_xent1_mse1000_lr1e-2-m=distill_llama3_1_405b_lk_smd_wtk64_fd64_w01-f=finetune_layer_mini_xent1_mse1000-s=0-se=0-re=0-ef=finetune_llama_405b_qkvo_e2-ft_lora=0-se=0-re=0-ef=finetune_llama_405b_qkvo_e2-ft_lora=0_epoch2.pt'

        # 3. RP Cria QV Rank 4 -- pure
        # PATH = '/home/rahul/code/lolcats/ckpts/cria_rp.pt'    # 780.pt step
        # PATH = '/home/rahul/code/lolcats/ckpt_lora-dl-d=rp_distill_llama_405b_xent1_mse1000_lr1e-2-m=distill_llama3_1_405b_lk_smd_wtk64_fd64_w01-f=rp_finetune_llama_40b_qv_hparams-s=0-se=0-re=0-ef=finetune_llama_405b_qkvo_e2_rp-ft_lora=0-se=0-re=0-s=1670.pt'

        print(f"PATH INFERENCE: {PATH}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        adapter_weights_path = os.getenv("LOLCATS_ADAPTER_PATH", PATH)
        adapter_weights = torch.load(adapter_weights_path, weights_only=True)

        adapter_weights_copy = OrderedDict({}) 

        for key, value in adapter_weights.items():
            key_suffix = key[key.rindex("model.")+6:]
            adapter_weights_copy[key_suffix] = value
        
        adapter_weights = adapter_weights_copy
        updated_keys = []

        print("\n")
        num_layers = len(self.model.layers)
        for layer_idx, layer in enumerate(self.model.layers):
            if layer_idx == 0:
                print(f'Weight factors before checkpoint load {self.tp_rank=}, {layer.self_attn.window_factors.shape}, {layer.self_attn.window_factors.flatten()}')

            window_factors_key = f'layers.{layer_idx}.self_attn.window_factors'
            if window_factors_key in adapter_weights:
                layer.self_attn.load_window_factors(adapter_weights[window_factors_key])
                updated_keys.append(window_factors_key)

                if layer_idx == 0:
                    print(f'Weight factors after checkpoint load {self.tp_rank=}, {layer.self_attn.window_factors.shape}, {layer.self_attn.window_factors.flatten()}')

            fm_q_key = f'layers.{layer_idx}.self_attn.feature_map_q.mlp.layer'
            if fm_q_key in adapter_weights:
                # if layer_idx in [0, num_layers-1]:
                #     # print("\n")
                #     # print(f'FMAP Q before checkpoint load {self.tp_rank=}, {layer.self_attn.feature_map_q.layer.shape}, {layer.self_attn.feature_map_q.layer[0,0,:4]}')

                layer.self_attn.load_feature_map_q(adapter_weights[fm_q_key])
                updated_keys.append(fm_q_key)

                # if layer_idx in [0, num_layers-1]:
                    # print(f'FMAP Q after checkpoint load; {layer.self_attn.feature_map_q.layer.shape},{layer.self_attn.feature_map_q.layer[0,0,:4]}')

            fm_k_key = f'layers.{layer_idx}.self_attn.feature_map_k.mlp.layer'
            if fm_k_key in adapter_weights:
                layer.self_attn.load_feature_map_k(adapter_weights[fm_k_key])
                updated_keys.append(fm_k_key)

            weight_name = 'layers.{layer_idx}.self_attn.{proj}.lora_{a_or_b}.default.weight'
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            # target_modules = ["q_proj", "k_proj", "v_proj"]
            # target_modules = ["k_proj", "v_proj"]
            # target_modules = ["q_proj", "k_proj"]

            r = 8
            lora_alpha = 16
            lora_dropout = 0

            for proj in target_modules:
                lora_A_key = weight_name.format(layer_idx=layer_idx, proj=proj, a_or_b='A')
                lora_B_key = weight_name.format(layer_idx=layer_idx, proj=proj, a_or_b='B')
                if lora_A_key in adapter_weights:
                    weight_A = adapter_weights[lora_A_key]
                    weight_B = adapter_weights[lora_B_key]
                    delta_AB = get_delta_weight(weight_A, weight_B, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                    
                    # if layer_idx in [0, num_layers-1]:
                    #     print(f'layer {layer_idx} weight_A.shape: {weight_A.shape} | weight_B.shape: {weight_B.shape} | delta_AB.shape: {delta_AB.shape}')
                    # print(f'layer {layer_idx} proj {proj} delta_AB', delta_AB.shape)
                    
                    if proj == 'o_proj':
                        # if layer_idx in [0, num_layers-1]:
                        #     print("\n")
                        #     print(f'Layer {layer_idx} {proj} weight before checkpoint load, {layer.self_attn.o_proj.weight.shape}, {layer.self_attn.o_proj.weight[0,:4]}')

                        layer.self_attn.merge_lora_to_o_parallel(delta_AB)

                        # if layer_idx in [0, num_layers-1]:
                        #     print(f'Layer {layer_idx} {proj} weight after checkpoint load, {layer.self_attn.o_proj.weight.shape}, {layer.self_attn.o_proj.weight[0,:4]}')
                    else:
                        # if layer_idx in [0, num_layers-1] and proj in ['q_proj']:
                        #     print(f'Layer {layer_idx} {proj} weight before checkpoint load, {layer.self_attn.qkv_proj.weight.shape}, {layer.self_attn.qkv_proj.weight[0,:4]}')

                        layer.self_attn.merge_lora_to_qkv_parallel(
                            delta_AB,
                            loaded_shard_id=proj.split('_')[0], 
                            total_num_heads=layer.self_attn.num_heads,
                            total_num_kv_heads=layer.self_attn.num_kv_heads,head_size=layer.self_attn.head_dim)

                        # if layer_idx in [0, num_layers-1] and proj in ['q_proj']:
                        #     print(f'Layer {layer_idx} {proj} weight after checkpoint load, {layer.self_attn.qkv_proj.weight.shape}, {layer.self_attn.qkv_proj.weight[0,:4]}')
                    updated_keys.append(lora_A_key)
                    updated_keys.append(lora_B_key)

        assert len(set(adapter_weights_copy.keys()) - set(updated_keys)) == 0, \
            f"UNUPDATED KEYS: {set(adapter_weights_copy.keys()) - set(updated_keys)}"


def transpose(weight, fan_in_fan_out):
    if not fan_in_fan_out:
        return weight

    if isinstance(weight, torch.nn.Parameter):
        return torch.nn.Parameter(weight.T)
    return weight.T


def get_delta_weight(weight_A: torch.Tensor, weight_B: torch.Tensor,                     
                     r: int = 8, lora_alpha: float = 16, lora_dropout: float = 0,
                     fan_in_fan_out: bool = False,):

    device = weight_B.device
    dtype = weight_B.dtype
    # From https://github.com/huggingface/peft/blob/850eeb5c3a5cf692f5612c7c733b13fde184e05d/src/peft/tuners/lora/layer.py#L512
    cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)
    if cast_to_fp32:
        weight_A = weight_A.float()
        weight_B = weight_B.float()
    scaling = lora_alpha / r
    output_tensor = transpose(weight_B @ weight_A, fan_in_fan_out) * scaling
    if cast_to_fp32:
        output_tensor = output_tensor.to(dtype=dtype)
    return output_tensor

