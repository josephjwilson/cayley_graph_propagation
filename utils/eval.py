import torch
import numpy as np

from tqdm import tqdm

from utils.config import cfg

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
    if cfg.metric == 'ACC':
        return _eval_acc(y_true, y_pred)
    else:
        raise ValueError('compute error')

def _eval_acc(y_true, y_pred):
    acc_list = []

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:,i] == y_true[:,i]
        correct = y_true[is_labeled,i] == y_pred[is_labeled,i]
        acc_list.append(float(np.sum(correct))/len(correct))

    result = sum(acc_list)/len(acc_list)

    return result
