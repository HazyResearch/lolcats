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

class LlamaLoraAttention(LlamaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _device = self.qkv_proj.weight.device
        _dtype = self.qkv_proj.weight.dtype
        print("Hello from Llama Lora Attention")

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
            torch.einsum('hdf,bhld->bhlf', self.layer, x))


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
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = num_heads // num_kv_heads

        max_seq_len = 2048
        window_size = 64

        self.register_buffer('mask_window', self._create_mask(max_seq_len, window_size, True))
        self.register_buffer('mask_linear', self._create_mask(max_seq_len, window_size, False))


    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        fmap_q: FeatureMap,
        fmap_k: FeatureMap,
        window_factors: torch.Tensor,
    ) -> torch.Tensor:
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

        f_q = fmap_q(query)
        f_k = fmap_k(key)

        window_size = 64
        window_factors = torch.nn.functional.sigmoid(window_factors)
        linear_factors = 1
        # linear_factors = 1 - window_factors

        return self.superlinear_attention(query, key, f_q, f_k, 
                                             value,
                                             window_factors, 
                                             linear_factors,
                                             window_size)
    
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
        return self.mask_window[:, :, :q_len, :k_len], self.mask_linear[:, :, :q_len, :k_len]

    def superlinear_attention(self, q: torch.Tensor, k: torch.Tensor, 
                                   f_q: torch.Tensor, f_k: torch.Tensor,
                                   v: torch.Tensor,
                                   window_factor: torch.Tensor,
                                   linear_factor: torch.Tensor,
                                   window_size: int,
                                   kv_state: torch.Tensor = None,
                                   k_state: torch.Tensor = None,
                                   eps: float = 1e-12,
                                   mask_value: float=-1e8):
        """
        Hybrid attention combining sliding window and linear attentions
        """

        mask_window, mask_linear = self.get_masks(
            window_size, q.shape[-2], k.shape[-2], q.device)

        # 1. Sliding window (softmax attention)
        # a_sm = torch.einsum(
        #     'bhmd,bhnd->bhmn', q.float(), k.float()) * (k.shape[-1] ** -0.5)
        a_sm = torch.einsum(
            'bhmd,bhnd->bhmn', q, k) * (k.shape[-1] ** -0.5)
        a_sm = a_sm.masked_fill(~mask_window.bool(), mask_value)
        # torch.softmax(a_sm, dim=-1), but we account for the max when combining
        a_sm_max = torch.amax(a_sm, dim=-1, keepdim=True)
        a_sm   = window_factor * torch.exp(a_sm - a_sm_max)
        sum_sm = a_sm.sum(dim=-1, keepdim=True)

        # 2. Under window (linear attention)
        # a_ln = torch.einsum('bhmd,bhnd->bhmn', f_q.float(), f_k.float())
        a_ln = torch.einsum('bhmd,bhnd->bhmn', f_q, f_k)
        a_ln = linear_factor * a_ln.masked_fill(~mask_linear.bool(), 0)
        sum_ln = a_ln.sum(dim=-1, keepdim=True)

        # 3. Combine
        # Allow outputs to also depend on prior kv_state and k_state
        # y = torch.einsum('bhmn,bhnd->bhmd', a_sm + a_ln, v.float())
        # y = (y / (sum_sm + sum_ln)).to(q.dtype)
        y = torch.einsum('bhmn,bhnd->bhmd', a_sm + a_ln, v)
        y = (y / (sum_sm + sum_ln))
        # # logger.info(f"splattn {y.shape=}")
        return y # attention weights only for the last chunk


class LlamaLolcatsAttention(LlamaAttention):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.attn = LlamaLolcatsAttentionActual(self.num_heads,
                                                self.head_dim,
                                                self.num_kv_heads)

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

        # print(f"{tp_size}")

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
    ) -> torch.Tensor:
        ndim = hidden_states.dim()
        qkv, _ = self.qkv_proj(hidden_states)
        seq_len = hidden_states.shape[-2]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v,
                                fmap_q=self.feature_map_q,
                                fmap_k=self.feature_map_k,
                                window_factors=self.window_factors)
        attn_output = attn_output.transpose(1, 2).contiguous().view(-1, seq_len, self.num_heads * self.head_dim)
        output, _ = self.o_proj(attn_output)
        if output.dim() > ndim:
            output = output.squeeze(0)
        return output


class LlamaLolcatsForCausalLM(LlamaForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"LOLCATS!!!: Loading model with config: {self.config}")

        softmax_attentions = getattr(self.config, 'softmax_attentions', [])

        for i in range(len(self.model.layers)):
            if i in softmax_attentions: 
                print(f"Using Lora Llama Attention at Layer {i}")
                self.model.layers[i].self_attn = LlamaLoraAttention(
                    config=self.config,
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_attention_heads,
                    num_kv_heads=getattr(self.config, "num_key_value_heads",
                                        self.config.num_attention_heads),
                    rope_theta=self.config.rope_theta,
                    rope_scaling=self.config.rope_scaling,
                )
            else:
                self.model.layers[i].self_attn = LlamaLolcatsAttention(
                    config=self.config,
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_attention_heads,
                    num_kv_heads=getattr(self.config, "num_key_value_heads",
                                        self.config.num_attention_heads),
                    rope_theta=self.config.rope_theta,
                    rope_scaling=self.config.rope_scaling,
                )

    def get_device(self):
        device = next(self.parameters()).device
        return str(device)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        super().load_weights(weights)

        # model_size = 8
        # model_size = 70
        model_size = 405

        # PATH = f'/data/rahul/checkpoints/{model_size}b.pt'
        # PATH = f'/home/rahul/code/lolcats/ckpt_lora-dl-d=distill_redpajama_xent1_mse1000_lr1e-2-m=distill_llama3_1_70b_lk_smd_wtk64_fd64_w01-f=finetune_lora_qkvo_redpajama-dcs=512-se=0-re=4-lzi=1-dcs=512-se=0-re=4.pt'
        
        # Trenchcoats v1
        # PATH = f'/home/rahul/code/lolcats/ckpt_lora-dl-d=distill_llama_405b_xent1_mse1000_lr1e-2-m=distill_llama3_1_405b_lk_smd_wtk64_fd64_w01-f=finetune_layer_mini_xent1_mse1000-s=0-se=0-re=0-ef=finetune_llama_405b_qkvo-ft_lora=0-se=0-re=0-ef=finetune_llama_405b_qkvo-ft_lora=0.pt'
        
        # No distill
        # PATH = f'/home/rahul/code/lolcats/ckpt_lora-dl-d=no_distill_alpaca_clean-m=distill_llama3_1_405b_lk_smd_wtk64_fd64_w01-f=finetune_layer_mini_xent1_mse1000-s=0-se=0-re=0-ef=no_distill_finetune_405b-ft_lora=0-se=0-re=0-ef=no_distill_finetune_405b-ft_lora=0-no_distill.pt'

        # Hybrid (last cria attention)
        PATH = f'/home/rahul/code/lolcats/ckpt_lora-dl-d=distill_llama_405b_xent1_mse1000_lr1e-2-m=distill_llama3_1_405b_lk_smd_wtk64_fd64_w01_h117-f=finetune_layer_mini_xent1_mse1000-s=0-se=0-re=0-ef=finetune_llama_405b_qkvo_cos-ft_lora=0-se=0-re=0-ef=finetune_llama_405b_qkvo_cos-ft_lora=0.pt'

        print(f"PATH: {PATH}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        adapter_weights_path = os.getenv("LOLCATS_ADAPTER_PATH", PATH)

        adapter_weights = torch.load(adapter_weights_path, weights_only=True)

        adapter_weights_copy = OrderedDict({})

        for key, value in adapter_weights.items():
            key_suffix = key[key.rindex("model.")+6:]
            adapter_weights_copy[key_suffix] = value
        
        adapter_weights = adapter_weights_copy
        updated_keys = []

        print("\n")
        for layer_idx, layer in enumerate(self.model.layers):
            # if layer_idx == 0:
            #     print(f'Weight factors before checkpoint load, {layer.self_attn.window_factors.shape}, {layer.self_attn.window_factors.flatten()}')

            window_factors_key = f'layers.{layer_idx}.self_attn.window_factors'
            if window_factors_key in adapter_weights:
                layer.self_attn.load_window_factors(adapter_weights[window_factors_key])
                updated_keys.append(window_factors_key)

                # if layer_idx == 0:
                #     print(f'Weight factors after checkpoint load, {layer.self_attn.window_factors.shape}, {layer.self_attn.window_factors.flatten()}')
            if layer_idx == 0:
                print("\n")
                print(f'FMAP Q before checkpoint load, {layer.self_attn.feature_map_q.layer.shape}, {layer.self_attn.feature_map_q.layer[0,0,:4]}')

            fm_q_key = f'layers.{layer_idx}.self_attn.feature_map_q.mlp.layer'
            if fm_q_key in adapter_weights:
                layer.self_attn.load_feature_map_q(adapter_weights[fm_q_key])
                updated_keys.append(fm_q_key)

                if layer_idx == 0:
                    print(f'FMAP Q after checkpoint load; {layer.self_attn.feature_map_q.layer.shape},{layer.self_attn.feature_map_q.layer[0,0,:4]}')

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
                    delta_AB = get_delta_weight(weight_A, weight_B, r=r, lora_alpha=lora_alpha, 
                                                lora_dropout=lora_dropout)
                    
                    # if layer_idx == 0:
                    #     print(f'layer {layer_idx} weight_A.shape: {weight_A.shape} | weight_B.shape: {weight_B.shape} | delta_AB.shape: {delta_AB.shape}')
                    # print(f'layer {layer_idx} proj {proj} delta_AB', delta_AB.shape)
                    
                    if proj == 'o_proj':
                        if layer_idx == 0:
                            print("\n")
                            print(f'Layer {layer_idx} {proj} weight before checkpoint load, {layer.self_attn.o_proj.weight.shape}, {layer.self_attn.o_proj.weight[0,:4]}')

                        layer.self_attn.merge_lora_to_o_parallel(delta_AB)

                        if layer_idx == 0:
                            print(f'Layer {layer_idx} {proj} weight after checkpoint load, {layer.self_attn.o_proj.weight.shape}, {layer.self_attn.o_proj.weight[0,:4]}')
                    else:
                        # if layer_idx == 0 and proj in ['q_proj']:
                        #     print(f'Layer {layer_idx} {proj} weight before checkpoint load, {layer.self_attn.qkv_proj.weight.shape}, {layer.self_attn.qkv_proj.weight[0,:4]}')

                        layer.self_attn.merge_lora_to_qkv_parallel(delta_AB,
                                                                   loaded_shard_id=proj.split('_')[0], 
                                                                   total_num_heads=layer.self_attn.num_heads,
                                                                   total_num_kv_heads=layer.self_attn.num_kv_heads,
                                                                   head_size=layer.self_attn.head_dim)
                        # if layer_idx == 0 and proj in ['q_proj']:
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
