"""
Combined dataloaders for Alpaca and CommonsenseQA
"""
from functools import partial
from os.path import join
import numpy as np

from torch.utils.data import Dataset
from datasets import load_metric, load_dataset

from .utils import (
    get_lm_loader, get_seq2seq_loader,
    convert_to_hf_dataset, 
    get_tokenizer_from_config,
    download_scrolls_metric as download_metric
)
from .utils.packing import ConcatDataset

from .alpaca_clean import load_data as load_data_alpaca
from .commonsense_qa import load_data as load_data_cqa


class CombinedConcatDataset(Dataset):
    def __init__(self, datasets: list[Dataset]):
        self.filtered_samples = []
        for dataset in datasets:
            self.filtered_samples.extend(dataset.filtered_samples)

    def __getitem__(self, idx):
        return self.filtered_samples[idx]
    
    def __len__(self):
        return len(self.filtered_samples)


def load_data(name: str, dataset_config: dict, pretrained_model_config: dict,
              preprocess_config: dict, **loader_kwargs: any):
    """
    Shared function to load dataset from experiment config
    -> e.g., see configs/experiments/distill_alpaca_clean_lr1e-2.yaml
    """
    input_len = dataset_config['alpaca']['chunk_size']
    dataloaders_alpaca = load_data_alpaca(name='alpaca',
                                          dataset_config=dataset_config['alpaca'],
                                          pretrained_model_config=pretrained_model_config,
                                          preprocess_config=preprocess_config,
                                          **loader_kwargs)
    dataloaders_cqa = load_data_cqa(name='commonsense_qa',
                                    dataset_config=dataset_config['commonsense_qa'],
                                    pretrained_model_config=pretrained_model_config,
                                    preprocess_config=preprocess_config,
                                    **loader_kwargs)

    datasets = {}
    for split in ['train', 'validation']:  # test split is not packed
        datasets[split] = CombinedConcatDataset([
            dataloaders_alpaca[split].dataset, dataloaders_cqa[split].dataset
        ])
    datasets['test'] = dataloaders_alpaca['test'].dataset

    tokenizer = dataloaders_alpaca[split].dataset.tokenizer   # see alpaca_clean.py

    # Get dataloaders
    dataloaders = {
        'train': get_lm_loader(datasets['train'], tokenizer, 'train', input_len, **loader_kwargs),
        'validation': get_lm_loader(datasets['validation'], tokenizer, 'validation', input_len, **loader_kwargs),
        'test': get_seq2seq_loader(datasets['test'], tokenizer, 'test', **loader_kwargs),
    }
    # Evaluation metric
    try:
        metric = load_metric(download_metric(), 'gov_report')  # hack for rouge
    except Exception as e:
        print(f'Error loading metric: {e}')
        metric = None

    # Finishing touches
    for k, v in dataloaders.items():  # Make tokenizer accessible
        dataloaders[k].dataset.tokenizer = tokenizer
        dataloaders[k].dataset.metric = metric
    return dataloaders
