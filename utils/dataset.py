from typing import List, Tuple, Optional, Union, Any, Dict, Callable, TypeVar, Generic, cast
from pathlib import Path

from utils.config import cfg

from torch_geometric.transforms import Compose, BaseTransform
from torch_geometric.datasets import TUDataset, LRGBDataset
from ogb.graphproppred import PygGraphPropPredDataset
from torch.utils.data import random_split, Subset, Dataset

from transforms.TuTransform import TuTransform
from transforms.PpaTransform import PpaTransform
from transforms.ExpanderTransform import ExpanderTransform
from transforms.FullyAdjacentTransform import FullyAdjacentTransform
from transforms.DiglTransform import DiglTransform
from transforms.SdrfTransform import SdrfTransform
from transforms.BorfTransform import BorfTransform
from transforms.GtrTransform import GtrTransform
from transforms.FosrTransform import FosrTransform

# Type aliases
T = TypeVar('T')
DatasetTuple = Tuple[Any, Any, Any, Optional[Any]]
DatasetLoader = Callable[[Optional[Compose]], DatasetTuple]

# Map of supported datasets
DATASET_MAP: Dict[str, Dict[str, Any]] = {
    'tu': {
        'names': ['MUTAG', 'ENZYMES', 'PROTEINS', 'COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY'],
        'loader': None,  # Will be set below
    },
    'lrgb': {
        'names': ['Peptides-func'],
        'loader': None,  # Will be set below
    },
    'ogb': {
        'names': ['ogbg-molhiv', 'ogbg-ppa'],
        'loader': None,  # Will be set below
    }
}

# Map of transform configurations
TRANSFORM_MAP: Dict[str, Dict[str, Any]] = {
    'egp': {
        'class': ExpanderTransform,
    },
    'cgp': {
        'class': ExpanderTransform,
    },
    'fa': {
        'class': FullyAdjacentTransform,
    },
    'digl': {
        'class': DiglTransform,
    },
    'sdrf': {
        'class': SdrfTransform,
    },
    'borf': {
        'class': BorfTransform,
    },
    'gtr': {
        'class': GtrTransform,
    },
    'fosr': {
        'class': FosrTransform,
    }
}

def load_datasets() -> DatasetTuple:
    """
    Load datasets based on configuration.
    
    Returns:
        Tuple containing train, validation, test, and complete datasets
    """
    # Create transforms
    transforms: Optional[Compose] = compose_transforms()
    
    # Determine dataset type and load accordingly
    dataset_format = cfg.dataset.format.lower()
    dataset_name = cfg.dataset.name
    
    # PyG datasets
    if dataset_format == 'pyg':
        # Check if dataset name is in TU datasets
        if dataset_name in DATASET_MAP['tu']['names']:
            return load_tu_dataset(transforms)
        # Check if dataset name is in LRGB datasets
        elif dataset_name in DATASET_MAP['lrgb']['names']:
            return load_lrgb_dataset(transforms)
        else:
            raise ValueError(f"Unknown PyG dataset: {dataset_name}. "
                            f"Supported TU datasets: {DATASET_MAP['tu']['names']}, "
                            f"Supported LRGB datasets: {DATASET_MAP['lrgb']['names']}")
    
    # OGB datasets
    elif dataset_format == 'ogb':
        if dataset_name in DATASET_MAP['ogb']['names']:
            return load_ogb_dataset(transforms)
        else:
            raise ValueError(f"Unknown OGB dataset: {dataset_name}. "
                            f"Supported OGB datasets: {DATASET_MAP['ogb']['names']}")
    
    # Unknown dataset format
    else:
        raise ValueError(f"Unknown dataset format: {dataset_format}. "
                        f"Supported formats: 'PyG', 'OGB'")

def load_tu_dataset(pre_transform: Optional[Compose] = None) -> DatasetTuple:
    """
    Load and prepare TUDataset, following (FoSR): https://arxiv.org/abs/2210.11790
    
    Args:
        pre_transform: Optional transform to apply to the dataset
        
    Returns:
        Tuple containing train, validation, test, and complete datasets
    """
    train_fraction: float = 0.8
    val_fraction: float = 0.1

    # Load dataset
    dataset: TUDataset = TUDataset(
        name=cfg.dataset.name, 
        root=make_dir_root(), 
        pre_transform=pre_transform
    )
    
    # Set model dimensions based on dataset
    set_input_dim_if_required(dataset.num_features)
    set_output_dim_if_required(dataset.num_classes)

    # Create train/val/test split
    dataset_size: int = len(dataset)
    train_size: int = int(train_fraction * dataset_size)
    validation_size: int = int(val_fraction * dataset_size)
    test_size: int = dataset_size - train_size - validation_size

    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, [train_size, validation_size, test_size]
    )

    return train_dataset, validation_dataset, test_dataset, dataset

def load_lrgb_dataset(pre_transform: Optional[Compose] = None) -> DatasetTuple:
    """
    Load and prepare LRGB dataset.
    
    Args:
        pre_transform: Optional transform to apply to the dataset
        
    Returns:
        Tuple containing train, validation, test datasets, and None for the complete dataset
    """
    # Create datasets for each split
    train_dataset: LRGBDataset = LRGBDataset(
        root=make_dir_root(), 
        name=cfg.dataset.name, 
        split="train", 
        pre_transform=pre_transform
    )
    
    validation_dataset: LRGBDataset = LRGBDataset(
        root=make_dir_root(), 
        name=cfg.dataset.name, 
        split="val", 
        pre_transform=pre_transform
    )
    
    test_dataset: LRGBDataset = LRGBDataset(
        root=make_dir_root(), 
        name=cfg.dataset.name, 
        split="test", 
        pre_transform=pre_transform
    )

    # Set model dimensions based on dataset
    set_output_dim_if_required(train_dataset.num_classes)

    return train_dataset, validation_dataset, test_dataset, None

def load_ogb_dataset(pre_transform: Optional[Compose] = None) -> DatasetTuple:
    """
    Load and prepare OGB dataset.
    
    Args:
        pre_transform: Optional transform to apply to the dataset
        
    Returns:
        Tuple containing train, validation, test, and complete datasets
    """
    # Load OGB dataset
    dataset: PygGraphPropPredDataset = PygGraphPropPredDataset(
        name=cfg.dataset.name, 
        root=make_dir_root(), 
        pre_transform=pre_transform
    )

    # Get train/val/test split indices
    split_idx = dataset.get_idx_split()
    
    # Set model dimensions based on dataset
    # Note: no need to set input_dim - OGB uses node encoder
    is_ppa = cfg.dataset.name.lower() == "ogbg-ppa"
    output_dim: int = dataset.num_classes if is_ppa else dataset.num_tasks
    set_output_dim_if_required(output_dim)

    # Create train/val/test datasets
    train_dataset = dataset[split_idx["train"]]
    validation_dataset = dataset[split_idx["valid"]]
    test_dataset = dataset[split_idx["test"]]

    return train_dataset, validation_dataset, test_dataset, dataset

def compose_transforms() -> Optional[Compose]:
    """
    Create a composition of dataset and graph transforms based on configuration.
    
    Returns:
        Compose object with transforms or None if no transforms are needed
    """
    transforms: List[BaseTransform] = []
    
    # Ensure dataset is selected
    if not cfg.dataset.name:
        raise ValueError("No dataset has been chosen")
    
    dataset_name = cfg.dataset.name.lower()
    
    # Add dataset-specific transforms
    if dataset_name in ['collab', 'reddit-binary', 'imdb-binary']:
        transforms.append(TuTransform())
    elif dataset_name == 'ogbg-ppa':
        transforms.append(PpaTransform())

    # Add network-specific transforms
    if cfg.transform.name is not None:
        transform_name = cfg.transform.name.lower()
        
        if transform_name not in TRANSFORM_MAP:
            available_transforms = list(TRANSFORM_MAP.keys())
            raise ValueError(f"Transform '{transform_name}' does not exist. "
                            f"Available options: {available_transforms}")
            
        # Create and configure the appropriate transform
        if transform_name == 'digl':
            # Get values from config with proper type handling
            alpha = cfg.transform.alpha if hasattr(cfg.transform, 'alpha') else 0.1
            k = cfg.transform.k if hasattr(cfg.transform, 'k') else 128
            eps_value = cfg.transform.eps if hasattr(cfg.transform, 'eps') else None
            
            # Pass eps as a keyword argument to work around the type inconsistency
            # The DiglTransform class has eps defined as float = None, which is inconsistent
            digl_kwargs = {'alpha': alpha, 'k': k}
            if eps_value is not None:
                digl_kwargs['eps'] = eps_value
                
            transforms.append(DiglTransform(**digl_kwargs))
        elif transform_name == 'sdrf':
            transforms.append(SdrfTransform(
                loops=cfg.transform.loops if hasattr(cfg.transform, 'loops') else 10,
                remove_edges=cfg.transform.remove_edges if hasattr(cfg.transform, 'remove_edges') else True,
                removal_bound=cfg.transform.removal_bound if hasattr(cfg.transform, 'removal_bound') else 0.5,
                tau=cfg.transform.tau if hasattr(cfg.transform, 'tau') else 1,
                is_undirected=cfg.transform.is_undirected if hasattr(cfg.transform, 'is_undirected') else False
            ))
        elif transform_name == 'borf':
            transforms.append(BorfTransform(
                loops=cfg.transform.loops if hasattr(cfg.transform, 'loops') else 10,
                remove_edges=cfg.transform.remove_edges if hasattr(cfg.transform, 'remove_edges') else True,
                removal_bound=cfg.transform.removal_bound if hasattr(cfg.transform, 'removal_bound') else 0.5,
                tau=cfg.transform.tau if hasattr(cfg.transform, 'tau') else 1,
                is_undirected=cfg.transform.is_undirected if hasattr(cfg.transform, 'is_undirected') else False,
                batch_add=cfg.transform.batch_add if hasattr(cfg.transform, 'batch_add') else 4,
                batch_remove=cfg.transform.batch_remove if hasattr(cfg.transform, 'batch_remove') else 2,
                algorithm=cfg.transform.algorithm if hasattr(cfg.transform, 'algorithm') else 'borf3'
            ))
        elif transform_name == 'gtr':
            transforms.append(GtrTransform(
                num_edges=cfg.transform.num_edges if hasattr(cfg.transform, 'num_edges') else 10,
                try_gpu=cfg.transform.try_gpu if hasattr(cfg.transform, 'try_gpu') else True
            ))
        elif transform_name == 'fosr':
            transforms.append(FosrTransform(
                num_iterations=cfg.transform.num_iterations if hasattr(cfg.transform, 'num_iterations') else 100,
                initial_power_iters=cfg.transform.initial_power_iters if hasattr(cfg.transform, 'initial_power_iters') else 10
            ))
        elif transform_name in ['egp', 'cgp']:
            transforms.append(ExpanderTransform())
        elif transform_name == 'fa':
            transforms.append(FullyAdjacentTransform())
    
    return None if not transforms else Compose(transforms)

def set_input_dim_if_required(input_dim: int) -> None:
    """
    Set the input_dim in config if not manually specified.
    
    Args:
        input_dim: The input dimension to set
    """
    if cfg.gnn.input_dim is None:
        cfg.gnn.input_dim = input_dim

def set_output_dim_if_required(output_dim: int) -> None:
    """
    Set the output_dim in config if not manually specified.
    
    Args:
        output_dim: The output dimension to set
    """
    if cfg.gnn.output_dim is None:
        cfg.gnn.output_dim = output_dim

def make_dir_root() -> str:
    """
    Create the directory path for dataset storage.
    
    Returns:
        Path string for the dataset storage
    """
    # Determine folder name based on dataset format/name
    if cfg.dataset.format.lower() == 'ogb':
        folder_name: str = 'ogb'
    elif cfg.dataset.name in DATASET_MAP['tu']['names']:
        folder_name: str = 'tu'
    elif cfg.dataset.name in DATASET_MAP['lrgb']['names']:
        folder_name: str = 'lrgb'
    else:
        raise ValueError(f"Unknown dataset format/name combination: {cfg.dataset.format}/{cfg.dataset.name}")

    # Determine transform name or use 'base' if none specified
    transform_name: str = 'base' if cfg.transform.name is None else cfg.transform.name.lower()

    # Create path string
    return f'{cfg.dataset.dir}/{folder_name}/{transform_name}'

# Add dataset loaders to map
DATASET_MAP['tu']['loader'] = load_tu_dataset
DATASET_MAP['lrgb']['loader'] = load_lrgb_dataset
DATASET_MAP['ogb']['loader'] = load_ogb_dataset
