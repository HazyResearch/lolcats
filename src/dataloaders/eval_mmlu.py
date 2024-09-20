"""
MMLU Evaluation Dataloader
"""
import sys
import collections
import random
import itertools

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



def get_mmlu_samples(task_dict_items: dict,
                     provide_description: bool = None,
                     num_fewshot: int = 5,
                     limit: int = None,
                     bootstrap_iters: int = 100000,
                     description_dict: dict = None,
                     check_integrity: bool = False,
                     decontamination_ngrams_path: str = None,
                     write_out: bool = False,
                     **kwargs: any):
    results = collections.defaultdict(dict)
    versions = collections.defaultdict(dict)
    
    requests = collections.defaultdict(list)
    requests_origin = collections.defaultdict(list)
    
    overlaps = collections.defaultdict(list)  # {task_name: contaminated_docs}
    
    docs = {}
    write_out_info = {}
    
    docs_for_decontamination = collections.defaultdict(list)
    
    decontaminate = decontamination_ngrams_path is not None
    
    # get lists of each type of request
    for task_name, task in task_dict_items:
        versions[task_name] = task.VERSION
        # default to test doc, fall back to val doc if validation unavailable
        # TODO: the test-fallback-to-val system isn't final, we should revisit it at some point
        if task.has_test_docs():
            task_doc_func = task.test_docs
            task_set = "test"  # Required for caching in the decontamination
        elif task.has_validation_docs():
            task_set = "val"  # Required for caching in the decontamination
            task_doc_func = task.validation_docs
        else:
            raise RuntimeError("Task has neither test_docs nor validation_docs")
    
        # deterministically shuffle docs and chop off the first `limit` because sometimes docs are in some kind of order
        task_docs = list(task_doc_func())
        rnd = random.Random()
        rnd.seed(42)
        rnd.shuffle(task_docs)
        print(f"Task: {task_name}; number of docs: {len(task_docs)}")
    
        if write_out:
            prompt_details = []
    
        description = (
            description_dict[task_name]
            if description_dict and task_name in description_dict
            else ""
        )
        if limit is not None:
            limit = int(len(task_docs) * limit) if limit < 1.0 else int(limit)
    
        for doc_id, doc in enumerate(itertools.islice(task_docs, 0, limit)):
            if decontaminate and task.should_decontaminate():
                docs_for_decontamination[(task_name, task_set)].append(
                    task.doc_to_decontamination_query(doc)
                )
    
            docs[(task_name, doc_id)] = doc
            ctx = task.fewshot_context(
                doc=doc, num_fewshot=num_fewshot, rnd=rnd, description=description
            )
            reqs = task.construct_requests(doc, ctx)
    
            if write_out:
                prompt_details.append({"doc_id": doc_id})
    
            # print the prompt for the first few documents
            if doc_id < 1:
                print(
                    f"Task: {task_name}; document {doc_id}; context prompt (starting on next line):\n{ctx}\n(end of prompt on previous line)"
                )
                print("Requests:", reqs)
    
            if not isinstance(reqs, (list, tuple)):
                reqs = [reqs]
            for i, req in enumerate(reqs):
                requests[req.request_type].append(req)
                # i: index in requests for a single task instance
                # doc_id: unique id that we can get back to a doc using `docs`
                requests_origin[req.request_type].append((i, task_name, doc, doc_id))
    
                if write_out:
                    prompt_details[-1][f"prompt_{i}"] = "".join(
                        (map(lambda x: "".join(x), req.args))
                    )
    
        if write_out:
            write_out_info[task_name] = prompt_details
    
    # Compare all tasks/sets at once to ensure a single training set scan
    if decontaminate:
        from lm_eval.decontamination.decontaminate import get_train_overlap
    
        print("Finding train/test overlap, please wait...")
        overlaps = get_train_overlap(
            docs_for_decontamination, decontamination_ngrams_path, limit
        )
    return requests, requests_origin, docs, versions
    


def load_data(name: str, dataset_config: dict, pretrained_model_config: dict,
              preprocess_config: dict, **loader_kwargs: any):
    """
    Shared function to load dataset from experiment config
    -> e.g., see configs/experiments/distill_alpaca_clean_lr1e-2.yaml
    """
    # Misc. setup
    cache_dir = dataset_config['cache_dir']

    # Tokenizer
    tokenizer_name = pretrained_model_config['pretrained_model_name_or_path']
    tokenizer_name = tokenizer_name.split('/')[-1]
    # save_path = join(cache_dir, f'{name}_{tokenizer_name}')
    
    # Setup tokenizer
    tokenizer = get_tokenizer_from_config(pretrained_model_config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f'Setting tokenizer.pad_token to {tokenizer.pad_token}')

    tokenizer.padding_side = 'left'  # for decoder-only generation

    # LM EVAL
    lm_eval_path = dataset_config['lm_evaluation_harness_path']  # '/juice2/scr2/mzhang/projects/lm-evaluation-harness'
    sys.path.append(lm_eval_path)

    # Load tasks
    from lm_eval.tasks import get_task_dict, TASK_REGISTRY
    if 'tasks' not in dataset_config:
        dataset_config['tasks'] = None
    tasks = dataset_config['tasks']
    print(f'tasks: {tasks}')
    
    if tasks is None:
        _task = 'hendrycksTest'
        tasks = sorted([k for k in TASK_REGISTRY.keys() if f'{_task}-' in k])
    else:
        tasks = sorted([k for k in TASK_REGISTRY.keys() if k in tasks])
    task_dict = get_task_dict(tasks)
    task_dict_items = [
        (name, task)
        for name, task in task_dict.items()
        if (task.has_validation_docs() or task.has_test_docs())
    ]

    # Prepare samples
    num_fewshot = dataset_config['num_fewshot']  # 5
    limit = dataset_config['limit']

    _samples = get_mmlu_samples(task_dict_items, num_fewshot=num_fewshot, limit=limit)
    
    requests, requests_origin, docs, versions = _samples
    requests = requests['loglikelihood']  # n-shot samples
    requests_origin = requests_origin['loglikelihood']  # Original sample
    # (0, 'mmlu-anatomy', {'query': 'Blood flows from the right ventricle of the heart into which of the following structures?\nA. Inferior vena cava\nB. Left ventricle\nC. Pulmonary arteries\nD. Pulmonary veins\nAnswer:', 'choices': ['A', 'B', 'C', 'D'], 'gold': 2}, 1)

    # breakpoint()
    # Get samples
    samples = [tokenizer(''.join(req.args)) for req in requests]  # ['loglikelihood']]
    for ix, sample in enumerate(samples):
        # sample_idx, category, query_dict, query_idx
        samples[ix]['target'] = requests_origin[ix][2]['gold']
        samples[ix]['query_idx'] = requests_origin[ix][-1]
        samples[ix]['category'] = tasks.index(requests_origin[ix][1])
    
    # for _ix, req in enumerate(requests):
    #     requests_origin[_ix][2]['text'] = ''.join(req.args)
    
    dataset = convert_to_hf_dataset(samples, cache_dir=cache_dir)
    if 'batch_size' in loader_kwargs:
        loader_kwargs['batch_size'] = 4  # for now enforce this
    dataloaders = {'eval': get_lm_loader(dataset, tokenizer, 'eval', **loader_kwargs)}

    # Finishing touches
    for k, v in dataloaders.items():  # Make tokenizer accessible
        dataloaders[k].dataset.tokenizer = tokenizer
        dataloaders[k].categories = tasks
    return dataloaders['eval']
