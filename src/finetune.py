"""
Finetuning functions to do post-distillation
"""
from os.path import join
from omegaconf import OmegaConf

import torch
from torch.nn import Module

from src.utils.setup import update_config_from_args
from src.dataloaders import load_data
from src.trainer import get_trainer, get_optimizer, get_scheduler


def prepare_finetune_configs(args, model_config: dict,
                             finetune_config_name: str = None,
                             finetune_checkpoint_name: str = None,
                             config_dir='./configs/experiment'):
    """
    Prepare finetuning configs
    """
    # Load finetuning config
    finetune_config = (finetune_config_name if finetune_config_name is not None else 
                       finetune_checkpoint_name.split('-f=')[-1].split('-')[0])
    finetune_config_path = join(config_dir, f'{finetune_config}.yaml')
    finetune_config = OmegaConf.load(finetune_config_path)
    finetune_config = update_config_from_args(finetune_config, args,
                                              ignore_args=['lr', 'weight_decay'])
    # Update data tokenizer to match model
    if getattr(finetune_config.dataset, 'pretrained_model_config', None) is not None:
        for k in ['pretrained_model_name_or_path', 'cache_dir']:
            finetune_config.dataset.pretrained_model_config[k] = model_config['model'][k]
    # Set finetuning args
    for arg, argv in finetune_config.trainer.items():
        if arg != 'name':
            setattr(args, arg, argv)
    for _config in ['dataloader', 'optimizer', 'lr_scheduler']:
        setattr(args, _config, OmegaConf.to_container(getattr(finetune_config, _config)))
    return finetune_config, args


def get_finetuner(model: Module, finetune_config: dict, device: torch.device, 
                  args: any, wandb: any, initial_eval: bool = False):
    """
    Initialize finetuning trainer
    """
    model.to(device)  # if using a fused optimizer
    model.train()

    # Initialize optimizer and scheduler
    optimizer = get_optimizer(model=model, **finetune_config.optimizer)
    scheduler = get_scheduler(optimizer=optimizer, **finetune_config.lr_scheduler)

    dataloaders  = load_data(finetune_config.dataset, finetune_config.dataloader)
    train_loader = dataloaders[finetune_config.trainer.train_split]
    eval_loader  = dataloaders[finetune_config.trainer.val_split]

    OurTrainer = get_trainer(finetune_config.trainer.name)
    trainer = OurTrainer(model=model,
                         args=args,
                         train_loader=train_loader,
                         eval_loader=eval_loader,
                         optimizer_and_scheduler=(optimizer, scheduler),
                         device=device,
                         wandb=wandb,
                         checkpoint_suffix='_ft',
                         **finetune_config.trainer)
    return trainer
