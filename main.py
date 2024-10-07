import time
import datetime

import torch
import numpy as np

from models.gnn import GNN

from utils.config import cfg
from utils.args import parse_args
from utils.config import cfg, set_cfg, load_cfg
from utils.loader import create_loaders
from utils.eval import eval_epoch
from utils.misc import create_loss_fn, create_optimiser, create_scheduler, params_count

from torch_geometric import seed_everything
from tqdm import tqdm

def train_epoch(train_loader, model, device, optimiser, loss_fn, scheduler):
    model.train()

    total_loss = 0

    for _, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        batch = batch.to(device)

        pred = model(batch)
        optimiser.zero_grad()

        is_labeled = batch.y == batch.y

        if cfg.metric == 'ACC':
            loss = loss_fn(pred.to(torch.float32)[is_labeled], batch.y[is_labeled])
        else:
            loss = loss_fn(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])

        total_loss += loss

        loss.backward()
        optimiser.step()

    if scheduler is not None:
        scheduler.step(total_loss)

def log_final_output(train_curve, validation_curve, test_curve, best_validation_epoch, epoch_times):
    print(f'Finished training: {datetime.datetime.now()}')
    print(f"Average time per epoch: {np.mean(epoch_times):.2f}s")
    print(f"Total train loop time: {np.sum(epoch_times) / 3600:.2f}h")

    #train_score = train_curve[best_validation_epoch]
    validation_score = validation_curve[best_validation_epoch]
    test_score = test_curve[best_validation_epoch]

    print('Best epoch {}'.format(best_validation_epoch + 1))
    print('Best validation score: {}'.format(validation_score))
    print('Test score: {}'.format(test_score))

def main():
    # Load config file and cmd line args
    args = parse_args()

    set_cfg(cfg)
    load_cfg(cfg, args)
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    device = torch.device(f'cuda:{cfg.device}') if torch.cuda.is_available() else torch.device('cpu')

    # Need to load before model, as it sets the input_dim/output_dim
    train_loader, validation_loader, test_loader, _ = create_loaders()

    model = GNN().to(device)

    loss_fn = create_loss_fn()
    optimiser = create_optimiser(model)
    scheduler = create_scheduler(optimiser)

    if cfg.train.loss_fn == 'cross_entropy':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif cfg.train.loss_fn == 'BCE':
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError('Loss function does not exist')

    print(f"Starting now: {datetime.datetime.now()} Seed: {cfg.seed}, dataset: {cfg.dataset.name }")
    print(model)
    print(cfg)
    cfg.params = params_count(model)
    print(f"Number of parameters: {cfg.params}")

    best_validation_epoch = 0
    best_validation_acc = 0.0

    train_curve = []
    validation_curve = []
    test_curve = []
    
    epoch_times = []

    has_stopping_criteria = cfg.train.stopping_patience > 0
    epochs_no_improvement = 0.0

    for epoch in range(1, 1 + cfg.optim.max_epochs):
        print("=====Epoch {}".format(epoch))
        print('Training...')

        start_time = time.perf_counter()

        train_epoch(train_loader, model, device, optimiser, loss_fn, scheduler)

        train_acc = eval_epoch(train_loader, model, device)
        validation_acc = eval_epoch(validation_loader, model, device)
        test_acc = eval_epoch(test_loader, model, device)

        epoch_times.append(time.perf_counter() - start_time)

        train_curve.append(train_acc)
        validation_curve.append(validation_acc)
        test_curve.append(test_acc)

        new_best_str = ''
        if validation_acc > best_validation_acc:
            best_validation_epoch = epoch - 1
            best_validation_acc = validation_acc            
            epochs_no_improvement = 0.0
            new_best_str = ' (new best validation)'
        elif validation_acc == best_validation_acc and test_acc > test_curve[best_validation_epoch]:
            # In line with: https://github.com/jeongwhanchoi/PANDA
            best_validation_epoch = epoch - 1
            epochs_no_improvement = 0.0
            new_best_str = ' (new best validation)'
        else:
            epochs_no_improvement += 1

        print(f'Train: {train_acc}, Validation: {validation_acc}{new_best_str}, Test: {test_acc}')

        if has_stopping_criteria and epochs_no_improvement > cfg.train.stopping_patience - 1:
            print(f'{cfg.train.stopping_patience} epochs without improvement, stopping training')
            break

    log_final_output(train_curve, validation_curve, test_curve, best_validation_epoch, epoch_times)
    
if __name__ == '__main__':
    main()
