import os
import torch
from os.path import join
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset


class AttentionInputDataset(Dataset):
    """
    Tensor dataset for LlamaAttention model
    """
    def __init__(self, tensors: torch.Tensor):
        self.samples = tensors

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        position_ids = torch.arange(x.shape[-2])
        return {'inputs_embeds': x, 'position_ids': position_ids}


# Data loader that we used for alpaca
def load_data_alpaca(data_dir: str, layer_idx: int, max_layer: int = 32, 
              **loader_kwargs: any):
    """
    Specific function to load attention input dataloaders
    """
    max_layer_digits = len(str(max_layer))

    dataloaders = {'train': None, 'validation': None}
    for split in dataloaders:
        sample_tensors = []
        
        for i, f in enumerate(tqdm(os.listdir(data_dir))):
            bs = int(f.split('-b=')[1].split('.')[0])
            if bs > 1400: 
                _act_split = "validation"
            else:
                _act_split = "train"

            # Filter and load naïvely 
            if f'-l={layer_idx:0{max_layer_digits}d}-s={split}' in f:
                print(f"Adding {f=}")
                sample_tensors.append(torch.load(join(data_dir, f)))
                
        samples = torch.cat(sample_tensors, dim=0)  # attn_inputs.shape is (batch, seq_len, hidden_size)
        _dataset = AttentionInputDataset(samples)
        _dataloader = DataLoader(_dataset, shuffle=True if split == 'train' else False,
                                 **loader_kwargs)
        dataloaders[split] = _dataloader
    return dataloaders


# red pajama
def load_data_redpajama(data_dir: str, layer_idx: int, max_layer: int = 32, 
              **loader_kwargs: any):
    """
    Specific function to load attention input dataloaders
    """
    max_layer_digits = len(str(max_layer))

    dataloaders = {'train': None, 'validation': None}
    train_sample_tensors = []
    val_sample_tensors = []
    for i, f in enumerate(tqdm(os.listdir(data_dir))):
        bs = int(f.split('-b=')[1].split('.')[0])

        # Filter and load naïvely 
        if f'-l={layer_idx:0{max_layer_digits}d}-s=train' in f:
            
            # for our crias
            end_train = 750
            end_val = end_train + 50

            if bs > end_train and bs < end_val: 
                try:
                    data = torch.load(join(data_dir, f))
                    val_sample_tensors.append(data)
                    print(f"Adding to val: {f=}")
                except:
                    print(f"Failed to load {f=}")
                    continue
            elif bs <= end_train:
                try:
                    data = torch.load(join(data_dir, f))
                    train_sample_tensors.append(data)
                    print(f"Adding to train: {f=}")
                except:
                    print(f"Failed to load {f=}")
                    continue
            else:
                continue

    print(f"{len(train_sample_tensors)=}, {len(val_sample_tensors)=}")

    # save train
    train_samples = torch.cat(train_sample_tensors, dim=0)  # attn_inputs.shape is (batch, seq_len, hidden_size)
    _dataset = AttentionInputDataset(train_samples)
    _dataloader = DataLoader(_dataset, shuffle=True, **loader_kwargs)
    dataloaders['train'] = _dataloader

    # save validation 
    val_samples = torch.cat(val_sample_tensors, dim=0)  # attn_inputs.shape is (batch, seq_len, hidden_size)
    _dataset = AttentionInputDataset(val_samples)
    _dataloader = DataLoader(_dataset, shuffle=False, **loader_kwargs)
    dataloaders['validation'] = _dataloader
    print(f"Got dataloaders!")
    return dataloaders


def load_data_redpajama_contig(data_dir: str, layer_idx: int, max_layer: int = 32, 
              **loader_kwargs: any):
    """
    Specific function to load attention input dataloaders
    """
    max_layer_digits = len(str(max_layer))

    samples = []
    for i, f in enumerate(tqdm(os.listdir(data_dir))):
        bs = int(f.split('-b=')[1].split('.')[0])

        # Filter and load naïvely 
        if f'-l={layer_idx:0{max_layer_digits}d}-s=train' in f:
            data = torch.load(join(data_dir, f))
            samples.append(data)
            print(f"Adding to train: {f=}")
        # if i > 10: break

    print(f"{len(samples)=}")

    # partition 
    num_train = int(len(samples) * 0.96)
    train_sample_tensors = samples[:num_train]
    val_sample_tensors = samples[num_train:]

    # save train
    train_samples = torch.cat(train_sample_tensors, dim=0)  
    train_dataset = AttentionInputDataset(train_samples)
    train_dataloader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)

    # save validation 
    val_samples = torch.cat(val_sample_tensors, dim=0)  
    val_dataset = AttentionInputDataset(val_samples)
    val_dataloader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    dataloaders = {'train': train_dataloader, 'validation': val_dataloader}
    return dataloaders



