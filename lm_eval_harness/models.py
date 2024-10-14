"""
Inherit from lm-evaluation-harness/lm_eval/models/huggingface.py to load linearized models
"""
from lm_eval.models.huggingface import AutoCausalLM
from src.model.modeling_llama import LolcatsLlamaForCausalLM as LOLCATS_LLAMA_MODEL_CLASS
from src.model.modeling_mistral import LolcatsMistralForCausalLM as LOLCATS_MISTRAL_MODEL_CLASS

from src.model.modeling_llama import LooooolcatsLlamaForCausalLM as LOOOOOLCATS_LLAMA_MODEL_CLASS
from src.model.modeling_mistral import LooooolcatsMistralForCausalLM as LOOOOOLCATS_MISTRAL_MODEL_CLASS

from src.model.modeling_llama_sharded import ShardedLolcatsLlamaForCausalLM as SHARDED_LOLCATS_LLAMA_MODEL_CLASS


class LolcatsLlamaForCausalLM(AutoCausalLM):
    """
    Wrapper for Llama-like autoregressive language model
    """
    AUTO_MODEL_CLASS = LOLCATS_LLAMA_MODEL_CLASS
    @property
    def add_special_tokens(self) -> bool:
        """Whether to include special tokens in encoded text. This should be
        determined by whether or not the model was trained with special tokens.
        TODO: Remove these conditionals once HuggingFace supports a way to
        check whether or not an arbitrary model was trained with special tokens.
        """
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        else:
            return False


class LolcatsMistralForCausalLM(AutoCausalLM):
    """
    Wrapper for Mistral-like autoregressive language model
    """
    AUTO_MODEL_CLASS = LOLCATS_MISTRAL_MODEL_CLASS
    @property
    def add_special_tokens(self) -> bool:
        """Whether to include special tokens in encoded text. This should be
        determined by whether or not the model was trained with special tokens.
        TODO: Remove these conditionals once HuggingFace supports a way to
        check whether or not an arbitrary model was trained with special tokens.
        """
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        else:
            return False


class ShardedLolcatsLlamaForCausalLM(AutoCausalLM):
    """
    Wrapper for Llama or Mistral-like autoregressive language model
    """
    AUTO_MODEL_CLASS = SHARDED_LOLCATS_LLAMA_MODEL_CLASS
    @property
    def add_special_tokens(self) -> bool:
        """Whether to include special tokens in encoded text. This should be
        determined by whether or not the model was trained with special tokens.
        TODO: Remove these conditionals once HuggingFace supports a way to
        check whether or not an arbitrary model was trained with special tokens.
        """
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        else:
            return False


# class ShardedRollLolcatsLlamaForCausalLM(AutoCausalLM):
#     """
#     Wrapper for Llama or Mistral-like autoregressive language model
#     """
#     AUTO_MODEL_CLASS = SHARDED_ROLL_LOLCATS_LLAMA_MODEL_CLASS
#     @property
#     def add_special_tokens(self) -> bool:
#         """Whether to include special tokens in encoded text. This should be
#         determined by whether or not the model was trained with special tokens.
#         TODO: Remove these conditionals once HuggingFace supports a way to
#         check whether or not an arbitrary model was trained with special tokens.
#         """
#         if self._add_special_tokens is not None:
#             return self._add_special_tokens
#         else:
#             return False
        

class LooooolcatsLlamaForCausalLM(AutoCausalLM):
    """
    Wrapper for Llama-like autoregressive language model
    """
    AUTO_MODEL_CLASS = LOOOOOLCATS_LLAMA_MODEL_CLASS
    @property
    def add_special_tokens(self) -> bool:
        """Whether to include special tokens in encoded text. This should be
        determined by whether or not the model was trained with special tokens.
        TODO: Remove these conditionals once HuggingFace supports a way to
        check whether or not an arbitrary model was trained with special tokens.
        """
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        else:
            return False


class LooooolcatsMistralForCausalLM(AutoCausalLM):
    """
    Wrapper for Mistral-like autoregressive language model
    """
    AUTO_MODEL_CLASS = LOOOOOLCATS_MISTRAL_MODEL_CLASS
    @property
    def add_special_tokens(self) -> bool:
        """Whether to include special tokens in encoded text. This should be
        determined by whether or not the model was trained with special tokens.
        TODO: Remove these conditionals once HuggingFace supports a way to
        check whether or not an arbitrary model was trained with special tokens.
        """
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        else:
            return False
