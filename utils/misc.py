# type: ignore

from typing import Optional, Union, Dict, Type, Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from utils.config import cfg

# Map of supported loss functions
LOSS_MAP: Dict[str, Type[nn.Module]] = {
    'cross_entropy': nn.CrossEntropyLoss,
    'bce': nn.BCEWithLogitsLoss,
}

# Map of supported optimizers
OPTIMISER_MAP: Dict[str, Type[Optimizer]] = {
    'adam': Adam,
}

# Map of supported schedulers
SCHEDULER_MAP: Dict[str, Callable] = {
    'reduce_on_plateau': ReduceLROnPlateau,
}

def create_loss_fn() -> nn.Module:
    """
    Create loss function based on configuration.
    
    Returns:
        PyTorch loss function
        
    Raises:
        ValueError: If specified loss function does not exist
    """
    loss_name = cfg.train.loss_fn.lower()
    
    if loss_name not in LOSS_MAP:
        available_losses = list(LOSS_MAP.keys())
        raise ValueError(f"Loss function '{loss_name}' is not supported. "
                         f"Available options: {available_losses}")
    
    return LOSS_MAP[loss_name]()

def create_optimiser(model: nn.Module) -> Optimizer:
    """
    Create optimizer for model training.
    
    Args:
        model: PyTorch model to optimize
        
    Returns:
        PyTorch optimizer
        
    Raises:
        ValueError: If specified optimizer does not exist
    """
    optimizer_name = cfg.optim.optimiser.lower()
    
    if optimizer_name not in OPTIMISER_MAP:
        available_optimizers = list(OPTIMISER_MAP.keys())
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported. "
                         f"Available options: {available_optimizers}")
    
    return OPTIMISER_MAP[optimizer_name](model.parameters(), lr=cfg.optim.base_lr)

def create_scheduler(optimizer: Optimizer) -> Optional[Union[_LRScheduler, ReduceLROnPlateau]]:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        PyTorch scheduler or None if not configured
        
    Raises:
        ValueError: If specified scheduler does not exist
    """
    if cfg.optim.scheduler is None:
        return None
    
    scheduler_name = cfg.optim.scheduler.lower()
    
    if scheduler_name not in SCHEDULER_MAP:
        available_schedulers = list(SCHEDULER_MAP.keys())
        raise ValueError(f"Scheduler '{scheduler_name}' is not supported. "
                         f"Available options: {available_schedulers}")

    scheduler_kwargs = {'optimizer': optimizer}
    if cfg.optim.scheduler_factor is not None:
        scheduler_kwargs['factor'] = cfg.optim.scheduler_factor
    if cfg.optim.scheduler_patience is not None:
        scheduler_kwargs['patience'] = cfg.optim.scheduler_patience
    if cfg.optim.scheduler_min_lr is not None:
        scheduler_kwargs['min_lr'] = cfg.optim.scheduler_min_lr

    # Create scheduler with appropriate parameters
    if scheduler_name == 'reduce_on_plateau':
        return SCHEDULER_MAP[scheduler_name](**scheduler_kwargs)
    
    return None

def params_count(model: nn.Module) -> int:
    """
    Compute the number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())
