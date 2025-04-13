from typing import List, Dict, Callable, Union, Optional, Any, cast
from functools import partial

import torch
import numpy as np

from tqdm import tqdm
from torch.nn import Module
from torch_geometric.loader import DataLoader

from utils.config import cfg
from sklearn.metrics import roc_auc_score

# Type aliases
MetricFunction = Callable[[np.ndarray, np.ndarray], float]

# Map of supported metrics - initializing as empty dict and filling later
METRIC_MAP: Dict[str, MetricFunction] = {}

def eval_epoch(loader: DataLoader, model: Module, device: torch.device) -> float:
    """
    Evaluate model performance for a single epoch.
    
    Args:
        loader: DataLoader containing batches to evaluate
        model: PyTorch model to evaluate
        device: Device to run evaluation on (CPU or GPU)
        
    Returns:
        float: Metric score (accuracy or ROC-AUC)
    """
    model.eval()
    y_true: List[torch.Tensor] = []
    y_pred: List[torch.Tensor] = []

    for _, batch in enumerate(tqdm(loader, desc="Evaluation")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)
        
        # Format predictions based on the metric type
        if cfg.metric.upper() == 'ACC':
            y_true.append(batch.y.view(-1, 1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())
        else:
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    # Convert predictions to numpy arrays
    y_true_np: np.ndarray = torch.cat(y_true, dim=0).numpy()
    y_pred_np: np.ndarray = torch.cat(y_pred, dim=0).numpy()

    return compute_metric(y_true_np, y_pred_np)

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the appropriate evaluation metric based on config settings.
    
    Args:
        y_true: Ground truth labels
        y_pred: Model predictions
        
    Returns:
        float: Computed metric value
        
    Raises:
        ValueError: If an invalid metric type is specified
    """
    metric = cfg.metric.upper()
    
    if metric not in METRIC_MAP:
        available_metrics = list(METRIC_MAP.keys())
        raise ValueError(f"Metric '{metric}' is not supported. "
                         f"Available options: {available_metrics}")
    
    return METRIC_MAP[metric](y_true, y_pred)

def _eval_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute accuracy metric averaged across tasks.
    
    Args:
        y_true: Ground truth labels
        y_pred: Model predictions
        
    Returns:
        float: Accuracy score
    """
    acc_list: List[float] = []

    for i in range(y_true.shape[1]):
        # Filter out NaN values
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))

    # Return average accuracy across tasks
    if not acc_list:
        raise RuntimeError('No valid data for accuracy calculation')
        
    return sum(acc_list) / len(acc_list)

def _eval_rocauc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute ROC-AUC metric averaged across tasks.
    
    Args:
        y_true: Ground truth labels
        y_pred: Model predictions
        
    Returns:
        float: ROC-AUC score
        
    Raises:
        RuntimeError: If no positively labeled data is available
    """
    rocauc_list: List[float] = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data
        # and at least one negative data
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # Ignore NaN values
            is_labeled = y_true[:, i] == y_true[:, i]
            # Cast the result to float to ensure type compatibility
            score = float(roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))
            rocauc_list.append(score)

    if not rocauc_list:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')
    
    return sum(rocauc_list) / len(rocauc_list)

# Register metric functions
METRIC_MAP['ACC'] = _eval_acc
METRIC_MAP['ROCAUC'] = _eval_rocauc

# Rename internal function to match public API
compute_metric = compute_metric  # Public API
