"""
Training loop helpers
"""
import torch
import numpy as np

from transformers.tokenization_utils import PreTrainedTokenizer


def replace_padding_tokens(token_ids: torch.Tensor, 
                           pad_token_id: int,
                           ignore_token_id: int = -100) -> any:
    """
    Replace ignore_token_id tokens with pad_token_id, 
    e.g., for printing inputs during training
    """
    if isinstance(token_ids, list):
        return [np.where(t != ignore_token_id, t, pad_token_id)[0] for t in token_ids]
    else:
        return np.where(token_ids != ignore_token_id, token_ids, pad_token_id)


def decode_samples(outputs: torch.Tensor,
                   targets: torch.Tensor,
                   tokenizer: PreTrainedTokenizer,
                   sample_idx: int = None) -> None:
    """
    Print first element of samples for debugging
    """
    print('=' * 20)
    print(f'*** TARGETS (sample {sample_idx})***')
    tokens = tokenizer.decode(
        replace_padding_tokens(targets[0], tokenizer.pad_token_id)
    )
    print(tokens)
    print('-' * 20)
    print(f'*** PREDICTIONS (sample {sample_idx}) ***')
    pred_logits = outputs.argmax(dim=-1).cpu()
    pred_tokens = tokenizer.decode(
        replace_padding_tokens(pred_logits[0], tokenizer.pad_token_id)
    )
    print(pred_tokens)
