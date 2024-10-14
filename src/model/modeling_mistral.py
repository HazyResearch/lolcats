# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
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
"""
Thin wrappers and replacement classes for MistralForCausalLM
"""
from typing import Optional, Tuple, List, Union

import warnings
import torch
import torch.nn as nn
from transformers import MistralModel, MistralForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from .modeling_llama import LolcatsLlamaModel
from .convert_model import get_attention_cache


# Modified from transformers.models.llama.modeling_llama.LlamaModel
class LolcatsMistralModel(LolcatsLlamaModel, MistralModel):
    """
    Wrapper for Mistral-like autoregressive language model
    """
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class LolcatsMistralForCausalLM(MistralForCausalLM):
    """
    Wrapper for Llama or Mistral-like autoregressive language model
    """
    def __init__(self, config):
        # Adapt config to LlamaConfig
        if getattr(config, 'attention_bias', None) is None:
            config.attention_bias = False
        if getattr(config, 'rope_scaling', None) is None:
            config.rope_scaling = None
        if getattr(config, 'pretraining_tp', None) is None:
            config.pretraining_tp = 1
        if getattr(config, 'pretraining_tp', None) is None:
            config.pretraining_tp = 1
        if getattr(config, 'mlp_bias', None) is None:
            config.mlp_bias = False
        super().__init__(config)
        self.model = LolcatsMistralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


class LooooolcatsMistralForCausalLM(LolcatsMistralForCausalLM):
    """
    Wrapper for Llama or Mistral-like autoregressive language model
    -> Experimental / WIP; but goal is to combine chunked linear attention during training
       to process long contexts with minimally-growing memory usage
    """
    def chunk_forward(self, *args: any, **kwargs: any):
        """Call this when training / processing one chunk"""
        return super().forward(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,  # Ignored for now, new Transformers >4.36
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass where we chunk inputs 
        """
        self.generating = False
        if use_cache is not True:
            use_cache = True
        
        if attention_mask is not None and use_cache:
            warnings.warn(
                f"Sorry padding currently not supported. Setting attention_mask to None (will still be causal)."
            )
            attention_mask = None

        if past_key_values is None:
            # Determine and setup our KV cache or state
            attention_type = getattr(self.model.layers[0].self_attn, 'attention_type', None)
            past_key_values = get_attention_cache(attention_type)
            # past_key_values = LinearAttentionState()

        if input_ids.shape[-1] == 1 and not self.training:  # Heuristic to detect generating
            return super().forward(input_ids, attention_mask, position_ids, 
                                   past_key_values, inputs_embeds, labels,
                                   use_cache, output_attentions, output_hidden_states,
                                   return_dict)
        else:
            if self.generating:  # Heuristic to detect new sample
                self.generating = False
                # Determine and setup our KV cache or state
                attention_type = getattr(self.model.layers[0].self_attn, 'attention_type', None)
                past_key_values = get_attention_cache(attention_type)
                print(f'-> attention_type:', attention_type)

            # Make it so we keep track of gradients in kv_state computation
            for idx in range(len(self.model.layers)):
                self.model.layers[idx].self_attn.state_grad_enabled = self.training

            # Split inputs into chunks, and do linear attention over each (passing the states)
            input_ids = torch.split(input_ids, self.state_chunk_len, dim=-1)
            if position_ids is not None:
                position_ids = torch.split(position_ids, self.state_chunk_len, dim=-1)

            all_logits = []  # save these
            for _idx, _input_ids in enumerate(input_ids):
                outputs = super().forward(_input_ids, None, 
                                          position_ids[_idx] if position_ids is not None else None, 
                                          past_key_values, inputs_embeds, 
                                          labels=None,
                                          use_cache=True, 
                                          output_attentions=False, 
                                          output_hidden_states=False,
                                          return_dict=True,)
                past_key_values = outputs.past_key_values
                all_logits.append(outputs.logits)

                # Comment in / adjust to do gradient accumulation over chunks
                # if self.training:
                #     loss = outputs.loss
                #     loss.backward()  # accumulate gradients over chunks
                # else:
                #     del outputs.loss

                if _idx == len(input_ids) - 1:
                    self.generating = True  # time to generate; if no generation will reset
                    
            return CausalLMOutputWithPast(
                # loss=loss,
                logits=torch.cat(all_logits, dim=-2),  # b, l, d
                past_key_values=past_key_values,
            )