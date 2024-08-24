# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json

import torch
from torch.utils.data import Dataset


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        self.ann = json.load(open(dataset_config.data_path))
        # Use 5% of the dataset for evaluation
        eval_length = int(len(self.ann)/20)
        if partition == "train":
            self.ann = self.ann[eval_length:]
        else:
            self.ann = self.ann[:eval_length]

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss


        ann = self.ann[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        example = prompt + ann["output"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1

        example_mask = example.ge(0)
        label_mask = labels.ge(0)

        # SA: Check for all -100 
        num_masked_elements = (~label_mask).sum().item()
        num_elements = (labels).sum().item()
        if (num_masked_elements == num_elements): 
            print(f"Warning: All {total_elements} elements in labels were initially masked. Randomly unmasking some elements.")
            num_to_unmask = 2
            all_indices = list(range(total_elements))
            indices_to_unmask = random.sample(all_indices, num_to_unmask)    
            for idx in indices_to_unmask:
                if idx < num_elements:
                    labels[idx] = example[idx+1]  # Set label to the corresponding example value
                    label_mask[idx] = True        # Update the mask
    
            print(f"Unmasked {num_to_unmask} elements.")
        
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }
