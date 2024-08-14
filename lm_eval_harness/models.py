"""
Inherit from lm-evaluation-harness/lm_eval/models/huggingface.py to load Hedgehog models
"""
from lm_eval.models.huggingface import AutoCausalLM
from src.models.modeling_llama import LolcatsLlamaForCausalLM as LOLCATS_LLAMA_MODEL_CLASS
from src.models.modeling_mistral import LolcatsMistralForCausalLM as LOLCATS_MISTRAL_MODEL_CLASS

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
