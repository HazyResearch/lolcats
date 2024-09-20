# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
"""
Copied from https://github.com/meta-llama/llama-recipes/blob/9b3dabcaac78980eae40005bbc8b1a8276c82af3/src/llama_recipes/data/concatenator.py#L1
"""
import random
from itertools import chain
from tqdm import tqdm


from torch.utils.data import Dataset


class ConcatContigDataset(Dataset):
    """
    Concatenates or packs samples of a dataset into chunks of size `chunk_size`
    -> But keep concatenated samples sourced within the same dataset
    """
    def __init__(self, dataset, chunk_size: int = 1024, seed: int = 42,) -> None:
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.samples = []
        random.seed(seed)
        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            buffer = {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }
            buffer = {k: v + sample[k] for k,v in buffer.items()}
            
            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}
        
        # Slow hack, but filter out any samples without valid labels (all -100)
        self.filtered_samples = []
        for s in self.samples:
            if sum(s['labels']) != chunk_size * -100:
                self.filtered_samples.append(s)
        if len(self.filtered_samples) < len(self.samples):
            print(f'OG dataset: {len(self.samples)} samples -> Filtered dataset: {len(self.filtered_samples)}')
            print(f'-> Filtered out {len(self.samples) - len(self.filtered_samples)} samples')
                
    def __getitem__(self, idx):
        return self.filtered_samples[idx]
    
    def __len__(self):
        return len(self.filtered_samples)
