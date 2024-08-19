"""
Alpaca Clean dataset with Llama3-Instruct prompt formatting
"""

from functools import partial
from os.path import join

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from datasets import load_metric, load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq, DefaultDataCollator, DataCollatorWithPadding

from .utils import (
    get_lm_loader, get_seq2seq_loader,
    convert_to_hf_dataset, 
    get_tokenizer_from_config,
    download_scrolls_metric as download_metric
)
from .utils.packing import ConcatDataset


SYSTEM_PROMPT = "You are a helpful AI assistant who always responds to appropriately complete a user's request."


def encode_response(response: str, tokenizer) -> list[int]:
    tokens = tokenizer.encode(response.strip(), add_special_tokens=False)
    # For Llama 3 Instruct: tokens.append(tokenizer.get_added_vocab()["<|eot_id|>"])
    tokens.append(tokenizer.eos_token_id)  
    try:  # Llama 3 Instruct
        tokens.append(tokenizer.get_added_vocab()["<|end_of_text|>"])
    except KeyError:
        pass
    return tokens


def load_data(name: str, dataset_config: dict, pretrained_model_config: dict,
              preprocess_config: dict, **loader_kwargs: any):

    # Misc. setup
    cache_dir = dataset_config['cache_dir']
    input_len = dataset_config['chunk_size']
    concat_data = dataset_config['concat_data']
    load_from_cache_file = False  # False if want to retokenize dataset

    # Hard-code system prompt handling
    if 'istral' in pretrained_model_config['pretrained_model_name_or_path']:
        system_prompt = ''
    else:
        system_prompt = SYSTEM_PROMPT

    tokenizer_name = pretrained_model_config['pretrained_model_name_or_path']
    tokenizer_name = tokenizer_name.split('/')[-1]
    save_path = join(cache_dir, f'{name}_{tokenizer_name}')
    
    # Setup tokenizer
    tokenizer = get_tokenizer_from_config(pretrained_model_config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f'Setting tokenizer.pad_token to {tokenizer.pad_token}')

    tokenizer.padding_side = 'left'  # for decoder-only generation

    # Get initial data
    ignore_kwargs = ['concat_data', 'chunk_size', 'pose_kwargs', 'system_prompt', 'name']
    train_set = load_dataset(
        **{k: v for k, v in dataset_config.items() if k not in ignore_kwargs},
        split='train[100:-100]',
    )
    val_set = load_dataset(  # we just use this dataset as a validation set
        **{k: v for k, v in dataset_config.items() if k not in ignore_kwargs},
        split='train[:100]+train[-100:]',
    )
    test_set = load_dataset(
        **{k: v for k, v in dataset_config.items() if k not in ignore_kwargs},
        split='train[:100]+train[-100:]',
    )

    # Convert to dicts of {input_ids, attention_mask, labels}
    train_set = train_set.map(partial(template_and_tokenize, tokenizer=tokenizer, 
                                      include_label=True, system_prompt=system_prompt),
                              remove_columns=list(train_set.features), 
                              load_from_cache_file=load_from_cache_file)
    val_set   = val_set.map(partial(template_and_tokenize, tokenizer=tokenizer, 
                                    include_label=True, system_prompt=system_prompt),
                            remove_columns=list(val_set.features),
                            load_from_cache_file=load_from_cache_file)
    test_set  = test_set.map(partial(template_and_tokenize, tokenizer=tokenizer, 
                                     include_label=False, system_prompt=system_prompt),
                             remove_columns=list(test_set.features),
                             load_from_cache_file=load_from_cache_file)

    # Chunk together train and val sets
    if concat_data:
        train_set = ConcatDataset(train_set, chunk_size=input_len)
        val_set = ConcatDataset(val_set, chunk_size=input_len)
    
    # Get dataloaders
    dataloaders = {
        'train': get_lm_loader(train_set, tokenizer, 'train', input_len, **loader_kwargs),
        'validation': get_lm_loader(val_set, tokenizer, 'validation', input_len, **loader_kwargs),
        'test': get_seq2seq_loader(test_set, tokenizer, 'test', **loader_kwargs),
    }
    # Evaluation metric
    metric = load_metric(download_metric(), 'gov_report')  # hack but we want rouge
    
    # Finishing touches
    for k, v in dataloaders.items():  # Make tokenizer accessible
        dataloaders[k].dataset.tokenizer = tokenizer
        dataloaders[k].dataset.metric = metric
    return dataloaders


def template_and_tokenize(sample, tokenizer, include_label: bool = True, 
                          system_prompt: str = None):
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    prompt = sample['instruction']
    if sample['input'] != '':
        prompt += f"\n\n{sample['input']}"
    
    messages = [
        {"role": "system", "content": system_prompt},
    ] if system_prompt != '' else []
    messages.append({"role": "user", "content": prompt})
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
    )
    if include_label:
        answer = encode_response(sample['output'], tokenizer)
    else:
        answer = []
        target = encode_response(sample['output'], tokenizer)
        
    input_ids = prompt_ids + answer
    attn_mask = [1] * len(input_ids)
    sample =  {
        "input_ids": input_ids,
        "attention_mask" : attn_mask,
        "labels": [-100] * len(prompt_ids) + answer if include_label else target,
    }
    return sample

