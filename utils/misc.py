import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.config import cfg

def create_loss_fn():
    if cfg.train.loss_fn == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()
    elif cfg.train.loss_fn == 'BCE':
        return torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError('Loss function does not exist')

def create_optimiser(model):
    if cfg.optim.optimiser == 'adam':
        return torch.optim.Adam(model.parameters(), lr=cfg.optim.base_lr)
    else:
        raise ValueError('Invalid optimiser')

def create_scheduler(optimiser):
    if cfg.optim.scheduler == 'reduce_on_plateau':
        return ReduceLROnPlateau(
            optimizer=optimiser,
            factor=cfg.optim.scheduler_factor,
            patience=cfg.optim.scheduler_patience,
            min_lr=cfg.optim.scheduler_min_lr
        )
    else:
        raise ValueError(f'Invalid optimiser: {cfg.optim.scheduler}')

def params_count(model):
    '''
    Computes the number of parameters.

    Args:
        model (nn.Module): PyTorch model

    '''
    return sum([p.numel() for p in model.parameters()])
