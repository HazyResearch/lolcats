# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""
Finetune attention-swapped model. Rough adaptation of llama_recipes script for distillation.
"""
import os
from os.path import join
import dataclasses
import random
import argparse  # ours
from pkg_resources import packaging

import torch
import torch.optim as optim

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing
from llama_recipes.utils.fsdp_utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (update_config,)
from llama_recipes.utils.fsdp_utils import ( hsdp_device_mesh as get_hsdp_device_mesh )
from llama_recipes.trainer_finetune import (
    train,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)
from llama_recipes.model_checkpointing.distill_checkpoint_handler import (
    load_model_sharded,
    load_sharded_model_single_gpu,
)
from accelerate.utils import is_xpu_available
from omegaconf import OmegaConf
from src.utils.setup import (
    update_config_from_args,
    update_model_config_from_args
)
from src.utils.logging import print_header, print_config
from src.finetune import prepare_finetune_configs  # get_finetuner
from src.model.pretrained import get_pretrained_loader
from distill_llama import (
    setup_wandb, get_args, 
    get_dataloaders, setup_fsdp_config
)
import torch.distributed as dist
from torch.distributed.elastic import rendezvous

from collections import defaultdict
from tqdm import tqdm


# Increase timeout
os.environ['TORCHELASTIC_MAX_WAIT_TIME'] = '600'  # 10 minutes

# Use a more robust rendezvous
rendezvous_config = rendezvous.RendezvousParameters(
    backend="c10d",
    endpoint="tcp://MASTER_ADDR:MASTER_PORT",
    run_id="MY_RUN_ID",
    min_nodes=1,  # Minimum number of nodes to wait for
    max_nodes=4,  # Maximum number of nodes to use
    timeout=600
)


def main():
    # ---------
    # 1. SET UP
    # ---------
    args = get_args()
    args.checkpoint_dir = join(args.checkpoint_dir, args.model_config)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    kwargs = vars(args)

    # Load distillation + attention configs
    distill_config_path = join('./configs/experiment', f'{args.distill_config}.yaml')
    distill_config = OmegaConf.load(distill_config_path)
    distill_config = update_config_from_args(distill_config, args)

    model_config_path = join('./configs/model', f'{args.model_config}.yaml')
    model_config = OmegaConf.load(model_config_path)
    model_config = update_model_config_from_args(model_config, args)
    if args.enable_fsdp:
        if getattr(model_config.model, 'load_in_4bit', False):
            model_config.model.device_map = 'auto'
        elif getattr(model_config.model, 'load_in_8bit', False):
            model_config.model.device_map = 'auto'
        else:
            model_config.model.device_map = None  # FSDP will complain about device placement o.w.

    # Update dataset pretrained model config
    for k in distill_config.dataset.pretrained_model_config:
        distill_config.dataset.pretrained_model_config[k] = getattr(model_config.model, k)

    args.run_name = args.run_name.replace('True', '1').replace('False', '0')  # concise hacks
    
    # Update the configuration for the training and sharding process
    distill_config = setup_fsdp_config(distill_config, args, 'distill')  # patch llama-recipes args
    fsdp_config = FSDP_CONFIG()
    update_config((fsdp_config), **vars(args))
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0 or not args.enable_fsdp:
        print_header('Distillation Config')
        print_config(distill_config)
        print_header('Model Config')
        print_config(model_config)
        print_header('FSDP Config')
        print_config(dataclasses.asdict(fsdp_config))

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # ------------------------
    # 2. LOAD PRETRAINED MODEL
    # ------------------------
    try:
        if not os.path.exists(model_config.model.pretrained_model_name_or_path):
            print(f"Model path {model_config.model.pretrained_model_name_or_path} does not exist. Using backup path. {model_config.model.pretrained_model_name_or_path_backup}")
            model_config.model.pretrained_model_name_or_path = model_config.model.pretrained_model_name_or_path_backup
        model_config.model.pop("pretrained_model_name_or_path_backup")
    except:
        print(f"Model without model.pretrained_model_name_or_path_backup path")
        pass

    model_loader = get_pretrained_loader(**model_config.model, 
                                         huggingface_token=args.huggingface_token)
    tokenizer = model_loader.load_tokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    use_cache = False if args.enable_fsdp else None

    if 'lama' in model_config.model.pretrained_model_name_or_path:
        from transformers import LlamaConfig as ModelConfig
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer as DecoderLayer
        from src.model.modeling_llama import LolcatsLlamaForCausalLM as ModelClass
        model_type = 'llama'

    # Convert model
    try:
        args.attention_type = model_config['attention']['attention_type']
    except AttributeError:
        args.attention_type = 'lolcats_llama'
    
    if args.enable_fsdp and args.low_cpu_fsdp:
        # for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        # this avoids cpu oom when loading large models like llama 70B, in which case
        # model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        # overhead and currently requires latest nightly.                
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly and rank == 0:
            print(f'-> Pytorch version is {v} ({v.dev})')
            print(f'   - Llama-recipes says "latest pytorch nightly build is required to run with low_cpu_fsdp config"')
            print(f"   - But who knows maybe this will work. We're just trying stuff.")
            print(f"   - (Also if PyTorch was installed after July 1, 2023 we should be good)")
        model = model_loader.load(args.attention_type)
        model.state_chunk_len = model_config['attention']['state_chunk_len']
    else:
        model = model_loader.load(args.attention_type)
        model.state_chunk_len = model_config['attention']['state_chunk_len']

    if rank == 0 or not args.enable_fsdp:
        print_header('Pretrained Model')
    
    model_config.model_name = model_config.model.pretrained_model_name_or_path
    print_model_size(model, model_config, rank if args.enable_fsdp else 0)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if args.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    if rank == 0:
        print_header('** Sanity check model weights **')
        for n, p in model.named_parameters():
            if ('layers.0.' in n and ('feature_map' in n or 'lora' in n)):
                print(f'-> {n}:\n', p)
            
    hsdp_device_mesh = None
    if fsdp_config.hsdp and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
        hsdp_device_mesh = get_hsdp_device_mesh(replica_group_size=fsdp_config.replica_group_size,
                                                sharding_group_size=fsdp_config.sharding_group_size)
        print("HSDP device mesh is ready")

    if args.enable_fsdp:
        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank, model=model_type)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, DecoderLayer)
        
        device_id = 0
        if is_xpu_available():
            device_id = torch.xpu.current_device()
        elif torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            print('-> device_id:', device_id)
       
        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrapping_policy,  # if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=args.low_cpu_fsdp,  # train_config.low_cpu_fsdp
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if args.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)

        # Load distilled checkpoints
        if args.verbose and rank == 0:
            print_header('*** FSDP Model ***')
            print(model)
            print('Loading checkpoints from:', distill_config.model_name)
                    
        if rank == 0 or not args.enable_fsdp:  # debugging
            print_header('** Sanity check model weights **')
            for n, p in model.named_parameters():
                if ('layers.0.' in n and 'base_attn' not in n and 
                    '.0.mlp.' not in n and '.block_sparse_moe' not in n):
                    print(f'-> {n}:\n', p)
        
    elif not model_config.model.quantization and not args.enable_fsdp:
        if is_xpu_available():
            model.to("xpu:0")
        elif torch.cuda.is_available():
            model.to("cuda")

    if args.verbose and (rank == 0 or not args.enable_fsdp):
        print_header('*** FSDP MODEL ***')
        print(model)
        print_header('*** Trainable Parameters ***')
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f'├── {n} (dtype = {p.dtype})')

    # Get data
    finetune_config, args = prepare_finetune_configs(args, model_config, 
                                                     args.finetune_config)
    # finetune_config = update_config_from_args(finetune_config, args)
    finetune_config = setup_fsdp_config(finetune_config, args, 'finetune')
    train_dataloader, eval_dataloader, finetune_config = get_dataloaders(finetune_config, tokenizer)
    if not args.enable_fsdp or rank == 0:
        print(f"--> Training   Set Length = {len(train_dataloader.dataset)}")
        print(f"--> Validation Set Length = {len(eval_dataloader.dataset)}")

    def get_outputs(model: torch.nn.Module, data: torch.Tensor, rank: int = 0, local_rank: int = 0):
        """Compute the loss for attention distillation"""
        input_keys = {'input_ids'}  # , 'attention_mask'} (assume packing / no padding)
        inputs = {k: v.to(model.device) for k, v in data.items() if k in input_keys}  
        outputs = model(**inputs, output_hidden_states=True, output_attentions=False, use_cache=False) 
        # outputs = outputs.get('logits')[..., :-1, :].contiguous()
        # targets = data.get('labels')[..., 1:].contiguous().detach().cpu()
        return outputs.hidden_states

    def gather_layer2dataset(layer2dataset, world_size, rank):
        gathered_layer2dataset = {}
        for layer, data_list in layer2dataset.items():
            gathered_data_list = []
            for data in data_list:
                gathered_tensors = [torch.zeros_like(data, device='cuda') for _ in range(world_size)] 
                dist.barrier()
                dist.all_gather(gathered_tensors, data.to(device='cuda'))
                dist.barrier()
                gathered_data_list.extend([tensor.cpu().detach() for tensor in gathered_tensors])
            if rank == 0:
                gathered_layer2dataset[layer] = gathered_data_list
        return gathered_layer2dataset


    layer2dataset = defaultdict(list)
    model_name = model_config.model['pretrained_model_name_or_path'].replace("/", "-")
    seqlen = distill_config.dataset['dataset_config']['chunk_size']
    if "8b" in model_name.lower(): interval = 100 
    elif "70b" in model_name.lower(): interval = 50
    else:
        assert 0, print("Unknown model.")

    for data_name, dataloader in [('train', train_dataloader), ('eval', eval_dataloader)]:
        layer2dataset_so_far = []
        pbar = tqdm(dataloader,colour="green", desc=f"Rank {rank}", dynamic_ncols=True)
        for step, batch in enumerate(pbar):
            for key in batch.keys():
                if args.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    if is_xpu_available():
                        batch[key] = batch[key].to('xpu:0')
                    else:
                        batch[key] = batch[key].to('cuda:0')
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                outputs = get_outputs(model, batch, rank=rank, local_rank=local_rank)
                for idx, hidden_state in enumerate(outputs[:-1]):  
                    hidden_state = model.model.layers[idx].input_layernorm(hidden_state).cpu()
                    layer2dataset[idx].append(hidden_state)
            pbar.set_description(f"Rank {rank}; Dataset size: {len(layer2dataset[0])}")
            
            # do it in chunks to shift stuff to the CPU
            if step % interval == 0 or step == len(pbar) - 1: 
                
                print(f"Synchronizing {step=}!")
                layer2dataset_collect = gather_layer2dataset(layer2dataset, world_size, rank)
                if rank == 0:
                    layer2dataset_so_far.append(layer2dataset_collect)
                # Clear the GPU memory after gathering
                layer2dataset.clear()
                torch.cuda.empty_cache()

                # Save the results so far
                if rank == 0: # Only the main process (rank 0) will save the data
                    combined_layer2dataset = {}
                    for layer_dict in layer2dataset_so_far:
                        for layer_idx, hidden_states in layer_dict.items():
                            if layer_idx not in combined_layer2dataset:
                                combined_layer2dataset[layer_idx] = []
                            combined_layer2dataset[layer_idx].extend(hidden_states)
                    save_dir="/scratch/sim/saved_output"
                    print(f"Rank {rank}; Num layers: {len(combined_layer2dataset)}; Dataset size: {len(combined_layer2dataset[0])}")
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{model_name}-{data_name}_seqlen{seqlen}-step{step}_layer2dataset.pt")
                    torch.save(combined_layer2dataset, save_path)
                    print(f"Saved synchronized {data_name}_layer2dataset to {save_path}")

                    # Clear CPU memory after saving
                    del combined_layer2dataset
                    del layer2dataset_so_far
                    import gc
                    gc.collect()
                layer2dataset_so_far = []

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
