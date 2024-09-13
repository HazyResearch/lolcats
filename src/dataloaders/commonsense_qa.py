"""
Alpaca training dataloaders

We adopt the original prompt template; goes something like:
```
Below is an instruction that describes a task. 
Write a response that appropriately completes the request.
### Instruction:
{instruction}
 
### Response:
{response}
```
See `PROMPT_DICT` for more. 
"""
from functools import partial
from os.path import join
import numpy as np

from datasets import load_metric, load_dataset

from .utils import (
    get_lm_loader, get_seq2seq_loader,
    convert_to_hf_dataset, 
    get_tokenizer_from_config,
    download_scrolls_metric as download_metric
)
from .utils.packing import ConcatDataset


def load_data(name: str, dataset_config: dict, pretrained_model_config: dict,
              preprocess_config: dict, **loader_kwargs: any):
    """
    Shared function to load dataset from experiment config
    -> e.g., see configs/experiments/distill_alpaca_clean_lr1e-2.yaml
    """
    # Misc. setup
    cache_dir = dataset_config['cache_dir']
    input_len = dataset_config['chunk_size']
    concat_data = dataset_config['concat_data']

    tokenizer_name = pretrained_model_config['pretrained_model_name_or_path']
    tokenizer_name = tokenizer_name.split('/')[-1]
    # save_path = join(cache_dir, f'{name}_{tokenizer_name}')
    
    # Setup tokenizer
    tokenizer = get_tokenizer_from_config(pretrained_model_config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f'Setting tokenizer.pad_token to {tokenizer.pad_token}')

    tokenizer.padding_side = 'left'  # for decoder-only generation
    # Get initial data
    loading_kwargs = ['path', 'cache_dir']
    dataset = load_dataset(
        **{k: v for k, v in dataset_config.items() if k in loading_kwargs}
    )

    # Preprocess samples into few-shot samples
    train_set = process_samples(dataset['train'], **dataset_config)
    val_set   = process_samples(dataset['validation'], **dataset_config)
    train_set = convert_to_hf_dataset(train_set, cache_dir=dataset_config['cache_dir'])
    _val_set  = convert_to_hf_dataset(val_set, cache_dir=dataset_config['cache_dir'])

    remove_columns = list(train_set.features)

    # Convert to dicts of {input_ids, attention_mask, labels}
    train_set = train_set.map(
        partial(template_and_tokenize, tokenizer=tokenizer, include_label=True), 
        remove_columns=remove_columns,) #  load_from_cache_file=False)
    val_set = _val_set.map(
        partial(template_and_tokenize, tokenizer=tokenizer, include_label=True),
        remove_columns=remove_columns,) #  load_from_cache_file=False)
    test_set = _val_set.map(  # This isn't the real test set
        partial(template_and_tokenize, tokenizer=tokenizer, include_label=False),
        remove_columns=remove_columns,)

    del _val_set

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
    try:
        metric = load_metric(download_metric(), 'qasper')  # hack but we want 
    except Exception as e:
        print(f'Error loading metric: {e}')
        metric = None

    # Finishing touches
    for k, v in dataloaders.items():  # Make tokenizer accessible
        dataloaders[k].dataset.tokenizer = tokenizer
        dataloaders[k].dataset.metric = metric
    return dataloaders


def process_samples(hf_dataset_split, num_shots: int, seed: int = 42, **kwargs: any):
    """
    Organize original dataset into few-shot sample datasets
    """
    fewshot_samples = []
    hf_dataset_split = hf_dataset_split.shuffle(seed=seed)
    context_counter = 0
    for i, _sample in enumerate(hf_dataset_split):
        if context_counter == 0:  # 5, (6, 7, 8, 9, 10)
            sample = {
                'context': [_sample],
                'question': [],
            }
            context_counter += 1
        elif context_counter % num_shots == 0:
            sample['question'] = _sample
            fewshot_samples.append(sample)
            context_counter = 0
        else:
            sample['context'].append(_sample)
            context_counter += 1
    return fewshot_samples


def template_and_tokenize(sample, tokenizer, include_label: bool = True):
    """
    Convert samples into text prompt and tokenize
    """
    prompt = ''
    # Add few-shot examples in context
    for ix, _sample in enumerate(sample['context']):
        prompt += _sample['question']
        for ix, label in enumerate(_sample['choices']['label']):
            text = _sample['choices']['text']
            prompt += f'\n{label}. {text[ix]}'
        prompt += f'\nAnswer: {_sample["answerKey"]}\n\n'
        # if ix < len(sample['context']) - 1:
        #     prompt += '\n\n'
    # Add question
    _sample = sample['question']
    prompt += _sample['question']
    for ix, label in enumerate(_sample['choices']['label']):
        text = _sample['choices']['text']
        prompt += f'\n{label}. {text[ix]}'
    prompt += f'\nAnswer: '
    prompt = tokenizer.encode(prompt, add_special_tokens=True)

    if include_label:
        answer = tokenizer.encode(f'{_sample["answerKey"]}{tokenizer.eos_token}', 
                                  add_special_tokens=False)
        target = None
    else:
        answer = []
        target = tokenizer.encode(f'{_sample["answerKey"]}{tokenizer.eos_token}', 
                                  add_special_tokens=False)
    input_ids = prompt + answer
    attn_mask = [1] * len(input_ids)

    sample =  {
        "input_ids": input_ids,
        "attention_mask" : attn_mask,
        "labels": [-100] * len(prompt) + answer if include_label else target,
    }
    return sample
