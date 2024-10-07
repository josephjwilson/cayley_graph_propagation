from utils.dataset import load_datasets
from utils.config import cfg

from torch_geometric.loader import DataLoader

def create_loaders():
    train_dataset, validation_dataset, test_dataset, complete_dataset = load_datasets()

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    complete_loader = DataLoader(complete_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader, complete_loader
