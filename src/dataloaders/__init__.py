"""
Load dataloaders
"""
import importlib


def load_data(dataset_config: dict, dataloader_config: dict):
    """Return dataloaders from dataset_config"""
    try:
        dataset_module = importlib.import_module(f'dataloaders.{dataset_config["name"]}')
    except Exception as e1:
        print(e1)
        try:  # e.g., tasks like GLUE where name is benchmark and path specifies the dataset / task
            dataset_module = importlib.import_module(f'dataloaders.{dataset_config["path"]}')
        except Exception as e2:
            print(f'Error from {dataset_config}')
            raise e2
    _load_data = getattr(dataset_module, 'load_data')
    return _load_data(**dataset_config, **dataloader_config)