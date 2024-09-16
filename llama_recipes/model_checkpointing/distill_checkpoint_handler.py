# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from pathlib import Path
from datetime import datetime
import time
import torch

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    LocalStateDictConfig,  # flattened params, usable only by FSDP
    ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)

from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    # FileSystemWriter,
    # save_state_dict,
    # load_state_dict,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
    # DefaultLoadPlanner,
)
import torch.distributed._shard.checkpoint as dist_cp
import torch.distributed as dist

# Added MZ 3/09/2024
from src.utils.logging import print_header


def _rename_sharded(n: str) -> str:
    """
    Rename sharded module names to match the original model
    """
    n = n.replace('_fsdp_wrapped_module.','')
    n = n.replace('._checkpoint_wrapped_module', '')
    n = n.replace('.mlp._flat_param', '.mlp.layer')  # feature_map
    n = n.replace('._flat_param', '.weight')
    return n


def get_trainable_weights(model: torch.nn.Module, keep_window_factors: bool = True) -> dict:
    """
    Get the state_dict of the model with only trainable parameters
    - state_dict() of FSDP-wrapped model collects weights
    """
    # Similar to:
    # return OrderedDict([
    #     (n, p.detach().cpu()) for n, p in model.named_parameters() if p.requires_grad
    # ])
    # But we still want to filter by params that require gradients
    state_dict = model.state_dict()
    save_params = [_rename_sharded(n) for n, p in model.named_parameters() if p.requires_grad]
    named_parameters = list(state_dict.keys())
    for n in named_parameters:
        if n not in save_params and ('window_factors' not in n or not keep_window_factors):  # hack
            del state_dict[n]
    return state_dict


def load_trainable_weights(model: torch.nn.Module, checkpoint: dict[any], rank: int):
    """
    Load trainable weights from a checkpoint to the model
    -> checkpoint weights are in `checkpoint['model']`
    """
    _keys = model.load_state_dict(checkpoint['model'], strict=False)
    if rank == 0:
        print_header('*** Keys loaded from state_dict ***')
        for k in checkpoint['model'].keys():
            print(k)
    try:
        assert len(_keys.unexpected_keys) == 0
        if rank == 0:
            print_header('*** All expected keys matched successfully ***')
    except AssertionError as e:
        if rank == 0:
            print(f'AssertionError: {e}')
            for n, p in model.named_parameters():
                if p.requires_grad:
                    print(n)
            print('=' * 20)
            print_header('*** Error: unexpected keys in checkpoint ***')
            print('Unexpected keys:')
            for k in _keys.unexpected_keys:
                print(k)
            print('=' * 20)
    return model


def get_date_of_run():
    """
    Create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run


# create singleton saving policies to avoid making over and over
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)


def load_model_sharded(model, rank, cfg, ignore_param_rule = None, model_path: str = None):
    
    # torch.manual_seed(103)
    if model_path is None:
        folder_name = (
            cfg.dist_checkpoint_root_folder
            + "/"
            + cfg.dist_checkpoint_folder
            + "-"
            + cfg.model_name
        )

        load_dir = Path.cwd() / folder_name

        if not load_dir.exists():
            if rank == 0:
                print(f"Error for {load_dir}:")
                print(f"-> No sharded_state_dict checkpoint directory found...skipping")
            return
        if rank == 0:
            print(f"loading model from model path: {load_dir} ")
    else:
        load_dir = Path(model_path)

    reader = FileSystemReader(load_dir)

    if ignore_param_rule is None:
        ignore_param_rule = lambda n, p: (
            not p.requires_grad   # and 'feature_map' not in n or ('v_proj' in n or 'o_proj' in n)
        )

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = model.state_dict()
        save_params = [
            _rename_sharded(n)
            for n, p in model.named_parameters() if not  ignore_param_rule(n, p)
        ]
        if rank == 0:
            print_header('xxx Ignored parameters xxx')
        named_parameters = list(state_dict.keys())
        for n in named_parameters:
            if n not in save_params and 'window_factors' not in n:  # hack
                if rank == 0:
                    print(n)
                del state_dict[n]
        checkpoint = {"model": state_dict}
        if rank == 0:
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
      
        dist_cp.load_state_dict(
            state_dict=checkpoint,
            storage_reader=reader,
        )
        if rank == 0:
            print("checkpoint after load_state_dict()")
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
        # model.load_state_dict(checkpoint["model"])
        model = load_trainable_weights(model, checkpoint, rank)
    if rank == 0:
        print(f"Sharded state checkpoint loaded from {load_dir}")


def save_model_and_optimizer_sharded(model, rank, cfg,optim=None):
    """save model and optimizer via sharded_state_dict to save_dir"""
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name
    )

    save_dir = Path.cwd() / folder_name
    if rank == 0:
        print(f"Saving model to {save_dir}")

    distributed_writer = dist_cp.FileSystemWriter(
        save_dir,
    )
    t0 = time.perf_counter()

    with FSDP.state_dict_type(model,
                              StateDictType.SHARDED_STATE_DICT,
                              ShardedStateDictConfig(offload_to_cpu=True),
                              ):
        # state_dict = {"model": model.state_dict()}
        state_dict = model.state_dict()
            
        # state_dict = model.state_dict(state_dict_device='cpu')
        save_params = [
            _rename_sharded(n)
            # n.replace('_fsdp_wrapped_module.','').replace('._checkpoint_wrapped_module', '').replace('.mlp._flat_param', '.mlp.layer').replace('._flat_param', '.weight')
            for n, p in model.named_parameters() if p.requires_grad
        ]
        named_parameters = list(state_dict.keys())
        for n in named_parameters:
            if n not in save_params and 'window_factors' not in n:  # hack
                del state_dict[n]
        # state_dict = {"model": get_trainable_weights(model)}
        state_dict = {"model": state_dict}
        if optim is not None:
            state_dict["optim"] = FSDP.optim_state_dict(model, optim)

        if rank == 0:
            for k, v in state_dict['model'].items():
                if 'layers.0' in k:
                    print(k, v.device)
        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=distributed_writer,
            planner=DefaultSavePlanner(),
            
        )
    dist.barrier()
    t1 = time.perf_counter()
    if rank == 0:
        print(f"Sharded state checkpoint saved to {save_dir}")
        print(
            f"Checkpoint Time = {t1-t0:.4f}\n"
        )
        get_date_of_run()
    return save_dir


def save_model_checkpoint(
    model,
    optimizer,
    rank,
    cfg,
    epoch=1,
):
    """saving model via rank0 cpu streaming and full_state_dict"""

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
    ):
        # cpu_state = model.state_dict()
        # cpu_state = get_trainable_weights(model)
        # trainable_weights(model)
        if rank == 0:
            print('Testing')
        state_dict = model.state_dict()
        save_params = [
            n.replace('_fsdp_wrapped_module.','').replace('._checkpoint_wrapped_module', '').replace('.mlp._flat_param', '.mlp.layer').replace('._flat_param', '.weight')
            for n, p in model.named_parameters() if p.requires_grad
        ]
        named_parameters = list(state_dict.keys())
        for n in named_parameters:
            if n not in save_params and 'window_factors' not in n:  # hack
                del state_dict[n]
        cpu_state = state_dict

        print(f"saving process: rank {rank}  done w model state_dict\n")
   

    if rank == 0:
        print("--> saving model ...")
        # create save path
        folder_name = (
        cfg.dist_checkpoint_root_folder
        )
        save_name = (
        cfg.model_name
        + "-"
        + cfg.dist_checkpoint_folder + ".pt"
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        # save_name = cfg.model_name + "-" + str(epoch) + ".pt"
        save_full_path = str(save_dir) + "/" + save_name

        # save model
        torch.save({"model": cpu_state}, save_full_path)
        # torch.save(cpu_state, save_full_path)
        
        print(f"model checkpoint saved for epoch {epoch} at {save_full_path}\n")
        return save_full_path
        

def load_model_checkpoint(model, rank, cfg):
    """load local checkpoint to rank0 cpu
    must be called * before * passing to FSDP"""

    if rank != 0:
        return

    # where is the checkpoint at...
    full_state_dict_model_path = (
        Path.cwd() / cfg.checkpoint_folder / cfg.checkpoint_model_filename
    )
    # is it present...
    if not full_state_dict_model_path.is_file():
        print(
            f"model checkpoint {full_state_dict_model_path} not present. Returning..."
        )
        return


    model_checkpoint = torch.load(full_state_dict_model_path)
    # integrate into loaded model
    # model.load_state_dict(model_checkpoint)
    model = load_trainable_weights(model, model_checkpoint, rank)
    print("model checkpoint loaded to rank0 cpu")


def save_optimizer_checkpoint(model, optimizer, rank, cfg, epoch=1):
    """save optimizer state via full state dict"""

    print(f"--> optim state call on rank {rank}\n")

    # pull all sharded optimizer states to rank0 cpu...
    optim_state = FSDP.full_optim_state_dict(model, optimizer)
    print(f"optim state dict ready on {rank} and len of {len(optim_state)}\n")

    if rank == 0:
        folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        opt_save_name = (
            "optimizer" + "-" + cfg.model_name + "-" + str(epoch) + ".pt"
        )
        opt_save_full_path = save_dir / opt_save_name

        print("--> saving optimizer state...")
        torch.save(optim_state, opt_save_full_path)
        print(f"--> saved {opt_save_full_path} to disk")


def load_optimizer_checkpoint(model, optimizer_checkpoint_path, rank):
    """load an fsdp optimizer full_state checkpoint using scatter method
    this ensures only rank 0 loads the optimizer state dict and scatters to other ranks
    """
    if not optimizer_checkpoint_path.is_file():
        print(
            f"warning - optimizer checkpoint not present {optimizer_checkpoint_path}. Returning. "
        )
        return

    full_osd = None

    if rank == 0:
        full_osd = torch.load(optimizer_checkpoint_path)

    # called from all ranks, though only rank0 has a valid param for full_osd
    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)
    print(f"optimizer shard loaded on rank {rank}")


def load_sharded_model_single_gpu(model, model_path=None, cfg=None, rank=None):
    """
    Load sharded model weights to a single model
    -> Should call this for the single model loaded on rank0 (which has actual weights)
    """
    if model_path is None:
        model_path = (
            cfg.dist_checkpoint_root_folder
            + "/"
            + cfg.dist_checkpoint_folder
            + "-"
            + cfg.model_name
        )

        model_path = Path.cwd() / model_path

        if not model_path.exists():
            if rank == 0:
                print(f"-> Error for {model_path}:")
                print("   -> No sharded_state_dict checkpoint directory found...skipping")
            return
        if rank == 0:
             print(f"loading model from model path: {model_path} ")
    # reader = FileSystemReader(model_path)
    # keep_window_factors = False if 'no_distill' in model_path else True
    keep_window_factors = True
    state_dict = {"model": get_trainable_weights(model, keep_window_factors=keep_window_factors)}
    print_header('*** (Trainable) keys in state_dict ***')
    for k, v in state_dict['model'].items():
        print(k)

    # breakpoint()
    dist_cp.load_state_dict(state_dict=state_dict, storage_reader= FileSystemReader(model_path), no_dist=True,)

    model = load_trainable_weights(model, state_dict, rank=0)
    print(f"Sharded state checkpoint loaded from {model_path}")
    return model
