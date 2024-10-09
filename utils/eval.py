import torch
import numpy as np

from tqdm import tqdm

from utils.config import cfg
from sklearn.metrics import roc_auc_score

def eval_epoch(loader, model, device):
    model.eval()
    y_true = []
    y_pred = []

    for _, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        if cfg.metric == 'ACC':
            y_true.append(batch.y.view(-1,1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())
        else:
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    return _compute_metric(y_true, y_pred)

def _compute_metric(y_true, y_pred):
    metric = cfg.metric.upper()
    if metric == 'ACC':
        return _eval_acc(y_true, y_pred)
    elif metric == 'ROCAUC':
        return _eval_rocauc(y_true, y_pred)
    else:
        raise ValueError(f'Invalid metric type: {metric}')

def _eval_acc(y_true, y_pred):
    acc_list = []

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:,i] == y_true[:,i]
        correct = y_true[is_labeled,i] == y_pred[is_labeled,i]
        acc_list.append(float(np.sum(correct))/len(correct))

    result = sum(acc_list)/len(acc_list)

    return result

def _eval_rocauc(y_true, y_pred):
    '''
        compute ROC-AUC averaged across tasks
    '''

    rocauc_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc_list.append(
                roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')
    
    result = sum(rocauc_list) / len(rocauc_list)

    return result
