from utils.config import cfg
from transforms.PpaTransform import PpaTransform

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import LRGBDataset
from ogb.graphproppred import PygGraphPropPredDataset
from torch.utils.data import random_split

tu_datasets = ['MUTAG', 'ENZYMES', 'PROTEINS', 'COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY']
lrgb_datasets = ['Peptides-func']

def load_datasets():
    if cfg.dataset.format == 'PyG':
        if cfg.dataset.name in tu_datasets:
            return load_tu_dataset()
        elif cfg.dataset.name in lrgb_datasets:
            raise ValueError('Dataset does not exist')
    elif cfg.dataset.format == 'OGB':
        pre_transform = PpaTransform() if cfg.dataset.name.lower() == "ogbg-ppa" else None
        return load_ogb_dataset(pre_transform)
    else:
        raise ValueError('Dataset does not exist')

def load_tu_dataset(pre_transform = None):
    # dataset in line with (FoSR): https://arxiv.org/abs/2210.11790
    train_fraction = 0.8
    val_fraction = 0.1

    dataset = TUDataset(name=cfg.dataset.name, root=make_dir_root(), pre_transform=pre_transform)
    
    set_input_dim_if_required(dataset.num_features)
    set_output_dim_if_required(dataset.num_classes)

    dataset_size = len(dataset)
    train_size = int(train_fraction * dataset_size)
    validation_size = int(val_fraction * dataset_size)
    test_size = dataset_size - train_size - validation_size

    train_dataset, validation_dataset, test_dataset = random_split(dataset,[train_size, validation_size, test_size])

    return train_dataset, validation_dataset, test_dataset, dataset

def load_lrgb_dataset(pre_transform = None):
    train_dataset = LRGBDataset(root=make_dir_root(), name=cfg.dataset.name, split="train", pre_transform=pre_transform)
    validation_dataset = LRGBDataset(root=make_dir_root(), name=cfg.dataset.name, split="val", pre_transform=pre_transform)
    test_dataset = LRGBDataset(root=make_dir_root(), name=cfg.dataset.name, split="test", pre_transform=pre_transform)

    set_output_dim_if_required(train_dataset.num_classes)

    return train_dataset, validation_dataset, test_dataset, None

def load_ogb_dataset(pre_transform = None):
    dataset = PygGraphPropPredDataset(name=cfg.dataset.name, root=make_dir_root(), pre_transform=pre_transform)

    split_idx = dataset.get_idx_split()
    
    # no need to set input_dim - OGB uses node encoder
    output_dim = dataset.num_classes if cfg.dataset.name.lower() == "ogbg-ppa" else dataset.num_tasks
    set_output_dim_if_required(output_dim)

    train_dataset = dataset[split_idx["train"]]
    validation_dataset = dataset[split_idx["valid"]]
    test_dataset = dataset[split_idx["test"]]

    return train_dataset, validation_dataset, test_dataset, dataset

def set_input_dim_if_required(input_dim):
    # Set the input_dim if not manually specified
    if cfg.gnn.input_dim is None:
        cfg.gnn.input_dim = input_dim

def set_output_dim_if_required(output_dim):
    # Set the output_dim if not manually specified
    if cfg.gnn.output_dim is None:
        cfg.gnn.output_dim = output_dim

def make_dir_root() -> str:
    if cfg.dataset.format == 'OGB':
        folder_name = 'ogb'
    elif cfg.dataset.name in tu_datasets:
        folder_name = 'tu'
    elif cfg.dataset.name in lrgb_datasets:
        folder_name = 'lrgb'
    else:
        raise ValueError('todo:')

    transform_name = 'base'

    return f'{cfg.dataset.dir}/{folder_name}/{transform_name}'
