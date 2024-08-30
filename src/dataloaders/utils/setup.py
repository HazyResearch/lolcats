"""
Helper functions dataset setup and loading
"""
import os
from os.path import join
import shutil
import numpy as np

from torch.utils.data import Dataset, DataLoader

from datasets import Dataset as HFDataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, LlamaTokenizer
from transformers import DataCollatorForSeq2Seq  
# from transformers import DefaultDataCollator, DataCollatorWithPadding


def get_seq2seq_loader(dataset: Dataset, tokenizer: AutoTokenizer,
                       split: str, **loader_kwargs: any):
    """
    Get dataloader for seq2seq tasks (evaluation)
    """
    tokenizer.padding_side = 'right'
    collate_fn = DataCollatorForSeq2Seq(
        tokenizer, label_pad_token_id=-100, return_tensors='pt')
    return DataLoader(
        dataset, shuffle='train' in split, collate_fn=collate_fn, **loader_kwargs)


def get_lm_loader(dataset: Dataset, tokenizer: AutoTokenizer,
                  split: str, max_length: int = None, **loader_kwargs: any):
    """
    Get dataloader for language modeling (training)
    -> Currently this ends up being the same as get_seq2seq_loader
    """
    # collate_fn = DefaultDataCollator(return_tensors='pt')
    # collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, padding=True,
    #                                      max_length=max_length, return_tensors='pt')
    collate_fn = DataCollatorForSeq2Seq(
        tokenizer, label_pad_token_id=-100, return_tensors='pt')
    return DataLoader(
        dataset, shuffle='train' in split, collate_fn=collate_fn, **loader_kwargs)


def convert_to_hf_dataset(dataset, cache_dir: str):
    """
    Convert iterable dataset to HuggingFace HFDataset object
    """
    def gen():
        for _, sample in enumerate(dataset):
            yield sample  # dataset[idx]
    return HFDataset.from_generator(gen, cache_dir=cache_dir)


def get_tokenizer_from_config(model_config):
    """
    Get pretrained tokenizer based on (pretrained) model config
    """
    # Get tokenizer
    if 'llama' in model_config['pretrained_model_name_or_path']:
        try:  # if we store locally
            model_path = join(model_config['cache_dir'],
                              model_config['pretrained_model_name_or_path'])
            tokenizer = LlamaTokenizer.from_pretrained(model_path)
        except Exception as e:
            try:
                tokenizer = AutoTokenizer.from_pretrained(**model_config)
                print("-> Bad LlamaTokenizer.from_pretrained(model_path)", e)
                print("-> But resolved with: AutoTokenizer.from_pretrained(**model_config)")
            except Exception as e2:
                print("-> Error with AutoTokenizer.from_pretrained(**model_config)", e2)
            # tokenizer = LlamaTokenizer.from_pretrained(**model_config)  # v4.43 errors with `*** TypeError: not a string`
    elif 'Mistral-7B-Instruct-v0.3' in model_config['pretrained_model_name_or_path']:
        tokenizer = LlamaTokenizer.from_pretrained(**model_config)  # hack where AutoTokenizer doesn't recognize
    elif 'Mistral-7B' in model_config['pretrained_model_name_or_path']:
        tokenizer = AutoTokenizer.from_pretrained(**model_config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(**model_config)
    return tokenizer


def add_special_tokens_to_dataset(dataset, tokenizer):
    """
    Add special tokens as attributes to a dataset object
    """
    token_map = {k: v for k, v in tokenizer.special_tokens_map.items()}
    special_ids = tokenizer.all_special_ids
    for idx, k in enumerate(tokenizer.special_tokens_map.keys()):
        token_map[f'{k}_id'] = special_ids[idx]
    for k, v in token_map.items():
        setattr(dataset, k, v)
    return dataset


def train_test_split(samples: any, train_size: int, test_size: int, seed: int):
    """
    Split samples into train and test sets
    """
    try:
        assert len(samples) == train_size + test_size
    except Exception as e:
        print(len(samples), train_size + test_size)
        raise e
    arange = np.arange(len(samples))
    np.random.seed(seed)
    test_idx = np.random.choice(arange, size=test_size, replace=False)
    train_idx = np.setdiff1d(arange, test_idx)
    return samples[train_idx], samples[test_idx]


def download_scrolls_metric():
    """
    Download ROUGE, F1, and other accuracy metrics included in the SCROLLS dataset
    """
    scrolls_metric_path = hf_hub_download(
        repo_id="tau/scrolls", filename="metrics/scrolls.py", repo_type="dataset"
    )
    updated_scrolls_metric_path = (
        os.path.dirname(scrolls_metric_path) + 
        os.path.basename(scrolls_metric_path).replace(".", "_") + ".py"
    )
    shutil.copy(scrolls_metric_path, updated_scrolls_metric_path)
    return updated_scrolls_metric_path
