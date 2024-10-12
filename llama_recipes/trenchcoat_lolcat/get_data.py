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


###################################################################
#  Use this if your data has explicit train/validation splits.
###################################################################
def load_data_alpaca(data_dir: str, layer_idx: int, max_layer: int = 32, 
              **loader_kwargs: any):
    """
    Specific function to load attention input dataloaders
    """

    train_batches_num = 1400

    max_layer_digits = len(str(max_layer))

    dataloaders = {'train': None, 'validation': None}
    for split in dataloaders:
        sample_tensors = []
        
        for i, f in enumerate(tqdm(os.listdir(data_dir))):
            bs = int(f.split('-b=')[1].split('.')[0])
            if bs > train_batches_num: 
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


###################################################################
#  Use these if your data has only train, and you need to split it.
###################################################################
def load_data_redpajama(
    data_dir: str, 
    layer_idx: int, 
    max_layer: int = 32, 
    **loader_kwargs: any
):
    """
    Specific function to load attention input dataloaders
    """
    max_layer_digits = len(str(max_layer))

    ####################################
    # SA 10/11/24: Set the number of batches to use for training and validation
    # these batches are obtained from the save_llama_attn_inputs.py script.
    # Note that we're just loading into CPU memory, so if you make this too 
    # large, you might run out of CPU memory.
    # #################################### 
    
    end_train = 2400
    end_val = end_train + 40

    dataloaders = {'train': None, 'validation': None}
    train_sample_tensors = []
    val_sample_tensors = []
    all_files = os.listdir(data_dir)
    print(f"Got {len(all_files)} files")
    for i, f in enumerate(tqdm(all_files)):
        bs = int(f.split('-b=')[1].split('.')[0])

        # Filter and load naïvely 
        if f'-l={layer_idx:0{max_layer_digits}d}-s=train' in f:

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
    print(f"{train_samples.shape=}")
    _dataset = AttentionInputDataset(train_samples)
    _dataloader = DataLoader(_dataset, shuffle=True, **loader_kwargs)
    dataloaders['train'] = _dataloader

    # save validation 
    val_samples = torch.cat(val_sample_tensors, dim=0)  # attn_inputs.shape is (batch, seq_len, hidden_size)
    print(f"{val_samples.shape=}")
    _dataset = AttentionInputDataset(val_samples)
    _dataloader = DataLoader(_dataset, shuffle=False, **loader_kwargs)
    dataloaders['validation'] = _dataloader
    print(f"Got dataloaders!")
    return dataloaders

