import importlib
from .optim import get_optimizer, get_scheduler


def get_trainer(name: str):
    """
    Return our trainer class
    """
    try:
        module = importlib.import_module(f'src.trainer.{name}')
    except ModuleNotFoundError as e:
        print(e)
        print('-> Using default trainer')
        module = importlib.import_module('src.trainer.default')
    return getattr(module, 'OurTrainer')
