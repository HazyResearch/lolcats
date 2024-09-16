"""
Data from https://github.com/FlagOpen/FlagEmbedding/blob/master/Long_LLM/longllm_qlora/src/data.py
-> Modifying code from above too
"""
from typing import Optional, List, Dict, Any, Mapping, Iterable, Union
from functools import partial
from os.path import join

from omegaconf import OmegaConf
from tqdm import tqdm

import os
import re
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import datasets
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq, DefaultDataCollator, DataCollatorWithPadding
from transformers.utils import logging

from .utils import get_tokenizer_from_config
from .utils.packing import ConcatDataset

logger = logging.get_logger(__name__)


def get_lm_loader(dataset: Dataset, tokenizer: AutoTokenizer, 
                  split: str, max_length: int = None, **loader_kwargs: any):
    """
    Get dataloader for language modeling (training)
    """
    collate_fn = DefaultDataCollator(return_tensors='pt')
    return DataLoader(dataset, shuffle='train' in split, 
                      collate_fn=collate_fn, **loader_kwargs)


def add_eos(inputs: Mapping, eos_token_id: int):
    """Add eos for BatchEncoding object."""
    assert isinstance(inputs["input_ids"], list), f"Make sure the return_tensors are set to list!"
    if inputs["input_ids"][-1] != eos_token_id:
        for k, v in inputs.items():
            if k in ["input_ids", "labels"]:
                v = v + [eos_token_id]
            elif k == "attention_mask":
                v = v + [1]
            elif k == "position_ids":
                v = v + [v[-1] + 1]
            elif k == "token_type_ids":
                v = v + v[-1:]
            else:
                raise NotImplementedError(f"Inputs key {k} not implemented!")
            inputs[k] = v
    return inputs


def load_data(name: str, dataset_config: dict, pretrained_model_config: dict,
              preprocess_config: dict, **loader_kwargs: any):
    dataset_config = OmegaConf.to_object(dataset_config)

    tokenizer_name = pretrained_model_config['pretrained_model_name_or_path']
    tokenizer_name = tokenizer_name.split('/')[-1]
    # save_path = join(cache_dir, f'{name}_{tokenizer_name}')
    
    # Setup tokenizer
    tokenizer = get_tokenizer_from_config(pretrained_model_config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f'Setting tokenizer.pad_token to {tokenizer.pad_token}')

    tokenizer.padding_side = 'left'  # for decoder-only generation
    # ^ But does this impact impact attention sink stuff?

    # Get initial data
    train_set = Data.prepare_train_data(
        dataset_config['train_data'], 
        tokenizer=tokenizer,
        max_length=dataset_config['max_length'],
        min_length=dataset_config['min_length'],
        chat_template=dataset_config['chat_template'],
        seed=dataset_config['seed'],
        cache_dir=dataset_config['cache_dir'],
    )
    val_set = Data.prepare_eval_data(
        dataset_config['eval_data'], 
        tokenizer=tokenizer,
        max_length=dataset_config['max_length'],
        min_length=dataset_config['min_length'],
        chat_template=dataset_config['chat_template'],
        seed=dataset_config['seed'],
        cache_dir=dataset_config['cache_dir'],
        max_eval_num=dataset_config['max_eval_num']
    )

    train_set = ConcatDataset(train_set, chunk_size=dataset_config['chunk_size'])
    val_set = ConcatDataset(val_set, chunk_size=dataset_config['chunk_size'])
    
    # Get dataloaders
    dataloaders = {
        'train': get_lm_loader(train_set, tokenizer, 'train', **loader_kwargs),
        'validation': get_lm_loader(val_set, tokenizer, 'validation', **loader_kwargs),
    }
    
    # Finishing touches
    for k, v in dataloaders.items():  # Make tokenizer accessible
        dataloaders[k].dataset.tokenizer = tokenizer
        # dataloaders[k].dataset.metric = metric
    return dataloaders


class Data:
    def _process_language_modeling(data, indices, tokenizer, min_length, max_length):
        outputs = {'input_ids': [], 'attention_mask': [], "labels": []} # , "length": [], "index": []}

        for i, text in enumerate(data['text']):
            # truncate text for faster processing
            encoded = tokenizer(text)
            if len(encoded["input_ids"]) < min_length:
                continue
            elif len(encoded['input_ids']) < max_length:
                encoded = add_eos(encoded, tokenizer.eos_token_id)
            else:
                for k, v in encoded.items():
                    encoded[k] = v[:max_length]

            encoded["labels"] = encoded["input_ids"].copy()

            for k, v in encoded.items():
                outputs[k].append(v)
        return outputs

    def prepare_train_data(data_files=None, tokenizer=None, max_length=4096, min_length=512, chat_template="llama-3", max_sample_num=None, seed=42, cache_dir=None, load_from_cache_file=None):
        if data_files is None:
            return None

        if isinstance(data_files, list):
            logger.info(f"Loading training data from {data_files}...")
        elif isinstance(data_files, str):
            logger.info(f"Loading training data from {data_files}...")
            data_files = [data_files]
        else:
            raise ValueError(f"Invalid training data {data_files}!")

        data_2_num_sample = {}
        for data_file in data_files:
            match = re.search("\[(\d*)\]", data_file)
            if match:
                max_sample_num = int(match.group(1))
                data_file = re.sub("\[(\d*)\]", "", data_file)
            else:
                max_sample_num = None
            data_2_num_sample[data_file] = max_sample_num   
        
        random.seed(seed)
        
        train_datasets = []
        for data_file, max_sample_num in data_2_num_sample.items():

            if os.path.isdir(data_file) and os.path.exists(os.path.join(data_file, "dataset_info.json")):
                # the dataset may be save_to_disk in advance
                dataset = datasets.load_from_disk(data_file)

            else:
                # the dataset is a json file
                data_file = os.path.join('/home/mzhang/projects/lolcats/data/long-llm', data_file)
                cache_dir = '/'.join(data_file.split('/')[:-1])
                print('cache_dir', cache_dir)
                dataset = datasets.load_dataset('json', data_files=data_file, split='train', cache_dir=cache_dir)

                column_names = dataset.column_names
                if "text" in column_names:
                    process_fn = partial(
                        Data._process_language_modeling, 
                        tokenizer=tokenizer, 
                        min_length=min_length, 
                        max_length=max_length
                    )
                elif "conversations" in column_names:
                    process_fn = partial(
                        Data._process_instruction_tuning, 
                        tokenizer=tokenizer, 
                        chat_template=chat_template, 
                        min_length=min_length, 
                        max_length=max_length
                    )
                else:
                    raise ValueError(f"Found neither 'text' nor 'conversations' in the training data!")

                dataset = dataset.map(process_fn, batched=True, num_proc=32, remove_columns=dataset.column_names, batch_size=32, with_indices=True, load_from_cache_file=load_from_cache_file)

            if max_sample_num is not None and len(dataset) > max_sample_num:
                dataset = dataset.train_test_split(max_sample_num, seed=seed)["test"]

            # index column is useless in training
            if "index" in dataset.column_names:
                dataset = dataset.remove_columns(["index"])

            train_datasets.append(dataset)

        dataset = datasets.concatenate_datasets(train_datasets)

        return dataset

    def prepare_eval_data(data_files=None, tokenizer=None, max_length=4096, min_length=512, chat_template="llama-3", max_eval_num=None, cache_dir=None, seed=42, load_from_cache_file=None):
        if data_files is None:
            return None

        random.seed(seed)

        data_files = os.path.join('/home/mzhang/projects/lolcats/data/long-llm', data_files[0])
        cache_dir = '/'.join(data_files.split('/')[:-1])
        print('cache_dir', cache_dir)

        if max_eval_num is not None:
            dataset = datasets.load_dataset('json', data_files=data_files, split=f'train[{-max_eval_num}:]', cache_dir=cache_dir)
        else:
            dataset = datasets.load_dataset('json', data_files=data_files, split='train', cache_dir=cache_dir)

        column_names = dataset.column_names
        if "text" in column_names:
            process_fn = partial(
                Data._process_language_modeling, 
                tokenizer=tokenizer, 
                min_length=min_length, 
                max_length=max_length
            )
        elif "conversations" in column_names:
            process_fn = partial(
                Data._process_instruction_tuning, 
                tokenizer=tokenizer, 
                chat_template=chat_template, 
                min_length=min_length, 
                max_length=max_length,
                eval_mode=True,
            )
        else:
            raise ValueError(f"Found neither 'text' nor 'conversations' in the training data!")

        dataset = dataset.map(process_fn, batched=True, num_proc=32, remove_columns=dataset.column_names, with_indices=True, load_from_cache_file=load_from_cache_file)
        return dataset
