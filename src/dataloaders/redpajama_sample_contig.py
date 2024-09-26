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
from .utils.packing_contig import ConcatContigDataset
from .redpajama_sample import Data

logger = logging.get_logger(__name__)

# Models for computing effective sequence length
from src.model.pretrained import get_pretrained_loader
from src.dataloaders.utils import get_tokenizer_from_config, convert_to_hf_dataset


def get_lm_loader(dataset: Dataset, shuffle: bool, **loader_kwargs: any):
    """
    Get dataloader for language modeling (training)
    """
    collate_fn = DefaultDataCollator(return_tensors='pt')
    return DataLoader(dataset, shuffle=shuffle, 
                      collate_fn=collate_fn, **loader_kwargs)


def load_data(name: str, dataset_config: dict, pretrained_model_config: dict,
              preprocess_config: dict, **loader_kwargs: any):
    dataset_config = OmegaConf.to_object(dataset_config)

    if 'num_train_samples' in dataset_config:
        num_train_samples = dataset_config['num_train_samples']
    else:
        raise NotImplementedError("Please include num_train_samples in under experiment_config.dataset.dataset_config")
    max_train_samples = dataset_config['max_train_samples']
    max_length = dataset_config['max_length']
    min_length = dataset_config['min_length']
    chunk_size = dataset_config['chunk_size']
    seed = dataset_config['seed']

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

    if 'load_from_cache_file' not in dataset_config:
        dataset_config['load_from_cache_file'] = None

    # Get initial data
    train_set = Data.prepare_train_data(
        dataset_config['train_data'], 
        tokenizer=tokenizer,
        max_length=dataset_config['max_length'],
        min_length=dataset_config['min_length'],
        chat_template=dataset_config['chat_template'],
        seed=dataset_config['seed'],
        cache_dir=dataset_config['cache_dir'],
        load_from_cache_file=dataset_config['load_from_cache_file']
    )
    val_set = Data.prepare_eval_data(
        dataset_config['eval_data'], 
        tokenizer=tokenizer,
        max_length=dataset_config['max_length'],
        min_length=dataset_config['min_length'],
        chat_template=dataset_config['chat_template'],
        seed=dataset_config['seed'],
        cache_dir=dataset_config['cache_dir'],
        max_eval_num=dataset_config['max_eval_num'],
        load_from_cache_file=dataset_config['load_from_cache_file']
    )

    train_set = ConcatContigDataset(train_set, chunk_size=dataset_config['chunk_size'])
    val_set   = ConcatContigDataset(val_set, chunk_size=dataset_config['chunk_size'])

    if 'filter_by_esl' not in dataset_config:
        dataset_config['filter_by_esl'] = False

    if 'filter_window' not in dataset_config:
        dataset_config['filter_window'] = 0
    window = dataset_config['filter_window']

    if dataset_config['filter_by_esl']:
        cache_dir = dataset_config['cache_dir']
        # train_set = convert_to_hf_dataset(train_set, cache_dir)
        # val_set = convert_to_hf_dataset(val_set, cache_dir)
    
        # Filter train_set for largest effective context length
        # -> Get precomputed topk samples
        _data_attr = dataset_config['train_data']
        _data_attr = '-d='.join(_data_attr).replace('/', '_').replace('.json', '')
        _data_attr = _data_attr.replace('[','_').replace(']','')
        
        # fname = f'd={_data_attr}-nts={num_train_samples}-mts={max_train_samples}-dcs={chunk_size}-max={max_length}-min={min_length}-s={seed}'
    
        
        try:
            fname = f'd={_data_attr}-mts={max_train_samples}-dcs={chunk_size}-max={max_length}-min={min_length}-s={seed}'
            fname = join(dataset_config['dataloaders_dir'], 'redpajama_sample_indices', fname)
            if dataset_config['filter_window'] > 0:
                sorted_idx = np.load(f'{fname}_l{window:03d}.npy')
            else:
                sorted_idx = np.load(f'{fname}.npy')
            sample_idx = sorted_idx[:num_train_samples]  # .numpy()
            train_set.filtered_samples = [train_set.filtered_samples[ix] for ix in sample_idx]
            print(f'-> Top {num_train_samples} indices loaded from {fname}!')
        except Exception as e:
            print(e)
            print(f'-> Error with loading from {fname}.npy. Computing...')
            model, model_config, tokenizer = load_model(config_dir='/scr-ssd/mzhang/projects/lolcats/configs',
                                                        model_config=dataset_config['esl_model_config'])
            model.eval()
            # Compute effective sequence lengths
            # -> each is shape: num_layers x num_heads x num_samples x seq_len
            train_loader = get_lm_loader(train_set, shuffle=False, **loader_kwargs)
            train_esl = compute_effective_seq_lens(model, train_loader, 
                                                   max_samples=max_train_samples)
            # Rank samples by effective sequence length
            _train_esl = train_esl.mean(0).mean(0).mean(-1)  # num_samples
            sorted_idx = torch.argsort(_train_esl, dim=-1, descending=True)
            # Save indices to generated filename
            fname = f'd={_data_attr}-mts={max_train_samples}-dcs={chunk_size}-max={max_length}-min={min_length}-s={seed}'
            fname = join(dataset_config['dataloaders_dir'], 'redpajama_sample_indices', fname)
            np.save(f'{fname}.npy', sorted_idx)
            print(f'-> Top {num_train_samples} saved to {fname}!')

            # Also sort by computing sequence lengths over last window tokens
            for window in [1, 2, 4, 8, 16, 32, 64, 128]:
                _train_esl = train_esl[..., -window:].mean(0).mean(0).mean(-1)  # num_samples
                sorted_idx = torch.argsort(_train_esl, dim=-1, descending=True)
                # Save indices to generated filename
                try:
                    _fname = f'{fname}_l{window:03d}.npy'
                    np.save(_fname, sorted_idx)
                    print(f'-> Samples saved to {_fname}!')

                    # Also save top samples
                    sample_idx = sorted_idx[:num_train_samples].numpy()
                    _fname = f'{fname}-nts={num_train_samples}_l{window:03d}.npy'
                    np.save(_fname, sample_idx)  # sorted_idx)
                    print(f'-> Top {num_train_samples} saved to {_fname}!')
                except:
                    sample_idx = sorted_idx[:num_train_samples].numpy()
                    _fname = f'{fname}-nts={num_train_samples}_l{window:03d}.npy'
                    np.save(_fname, sample_idx)  # sorted_idx)
                    print(f'-> Top {num_train_samples} saved to {_fname}!')
            
            sample_idx = sorted_idx[:num_train_samples].numpy()
            train_set.filtered_samples = [train_set.filtered_samples[ix] for ix in sample_idx]
    
    # Get dataloaders
    dataloaders = {
        'train': get_lm_loader(train_set, shuffle=True, **loader_kwargs),
        'validation': get_lm_loader(val_set, shuffle=False, **loader_kwargs),
    }
    
    # Finishing touches
    for k, v in dataloaders.items():  # Make tokenizer accessible
        dataloaders[k].dataset.tokenizer = tokenizer
    return dataloaders


# Helpers for computing longest effective sequence length samples

def load_model(config_dir: str = './configs',
               model_config: str = 'base_llama3_8b'):
    """
    Load pretrained LLM (default is Llama 3 8B)
    """
    model_config = join(config_dir, 'model', f'{model_config}.yaml')
    model_config = OmegaConf.load(model_config)
    model_config['model']['attn_implementation'] = 'eager'  # for attentions
    
    # Load initial model
    model_loader = get_pretrained_loader(**model_config.model)
    tokenizer = model_loader.load_tokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    
    model = model_loader.load(model_config['attention']['attention_type'])
    return model, model_config, tokenizer


def compute_effective_seq_lens(model, dataloader, max_samples: int = None):
    """Compute effective sequence length for each sample"""
    effective_seq_len_by_layer_by_head = [[] for _ in range(len(model.model.layers))]
    batch_idx = 0  # always batch size 1
    with torch.no_grad():
        for ix, data in enumerate(tqdm(dataloader, desc='Computing effective sequence lengths')):
            inputs = {k: v.to(model.device) for k, v in data.items() if k != 'labels'}
            outputs = model(**inputs, output_attentions=True, use_cache=False, 
                            output_hidden_states=False)
            outputs = outputs.get('attentions')
            a_true_by_layer = outputs
            for layer_idx in range(len(a_true_by_layer)):
                # Effective seq len
                _true_attns = a_true_by_layer[layer_idx][batch_idx]  # num_heads x seq_len x seq_len
                positions = torch.arange(_true_attns.shape[-1], device=_true_attns.device)
                distances = positions[:, None] - positions[None, :]  # "outer diff"
                effective_seq_len = (distances * _true_attns).sum(dim=-1).cpu()
                effective_seq_len_by_layer_by_head[layer_idx].append(effective_seq_len)
            del a_true_by_layer; del positions; del distances
            if max_samples is not None:
                if ix + 1 == max_samples:
                    break
    
    esl = torch.stack([
        torch.stack(effective_seq_len_by_layer_by_head[_idx])
        for _idx in range(len(effective_seq_len_by_layer_by_head))
    ])
    esl = esl.transpose(2, 1)  # num_layers x num_heads x num_samples x seq_len
    return esl
