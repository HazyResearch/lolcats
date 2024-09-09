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

from datasets import load_metric, load_dataset

from .utils import (
    get_lm_loader, get_seq2seq_loader,
    convert_to_hf_dataset, 
    get_tokenizer_from_config,
    download_scrolls_metric as download_metric
)
from .utils.packing import ConcatDataset


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}


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
    ignore_kwargs = ['concat_data', 'chunk_size', 'pose_kwargs']
    dataset = load_dataset(
        **{k: v for k, v in dataset_config.items() if k not in ignore_kwargs}
    )
    if dataset_config['name'] == 'samsum':  # hack
        dataset = dataset.rename_column('dialogue', 'input')
        dataset = dataset.rename_column('summary', 'output')
        _instruction = 'Summarize this dialogue.'
        for split in dataset.keys():
            dataset[split] = dataset[split].add_column(
                'instruction', [_instruction] * len(dataset[split])
            )
        train_set, val_set, test_set = dataset['train'], dataset['validation'], dataset['test']
        dataset = train_set  # hack to work with below code
    else:
        dataset = dataset['train']
        train_set = convert_to_hf_dataset([dataset[ix] for ix in range(200, len(dataset))], cache_dir)
        val_set   = convert_to_hf_dataset([dataset[ix] for ix in range(200)], cache_dir)
        test_set  = convert_to_hf_dataset([dataset[ix] for ix in range(200)], cache_dir)

    # Convert to dicts of {input_ids, attention_mask, labels}
    train_set = train_set.map(
        partial(template_and_tokenize, tokenizer=tokenizer, include_label=True), 
        remove_columns=list(dataset.features),) #  load_from_cache_file=False)
    val_set = val_set.map(
        partial(template_and_tokenize, tokenizer=tokenizer, include_label=True),
        remove_columns=list(dataset.features),) #  load_from_cache_file=False)
    test_set  = test_set.map(
        partial(template_and_tokenize, tokenizer=tokenizer, include_label=False),
        remove_columns=list(dataset.features),) #  load_from_cache_file=False)

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
        metric = load_metric(download_metric(), 'gov_report')  # hack but we want rouge
    except Exception as e:
        print(f'Error loading metric: {e}')
        metric = None

    # Finishing touches
    for k, v in dataloaders.items():  # Make tokenizer accessible
        dataloaders[k].dataset.tokenizer = tokenizer
        dataloaders[k].dataset.metric = metric
    return dataloaders


def template_and_tokenize(sample, tokenizer, include_label: bool = True):
    """
    Format dataset context and answers into single-sequence prompts
    """
    if sample.get('input', '') == '':
        prompt = PROMPT_DICT["prompt_no_input"].format_map(sample)
    else:
        prompt = PROMPT_DICT["prompt_input"].format_map(sample)

    prompt = tokenizer.encode(prompt, add_special_tokens=True)
    if include_label:
        answer = tokenizer.encode(f'{sample["output"]}{tokenizer.eos_token}', 
                                  add_special_tokens=False)
        target = None
    else:
        answer = []
        target = tokenizer.encode(f'{sample["output"]}{tokenizer.eos_token}', 
                                  add_special_tokens=False)
    input_ids = prompt + answer
    attn_mask = [1] * len(input_ids)

    sample =  {
        "input_ids": input_ids,
        "attention_mask" : attn_mask,
        "labels": [-100] * len(prompt) + answer if include_label else target,
    }
    return sample
