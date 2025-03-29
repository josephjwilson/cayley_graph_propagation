from typing import Tuple, Dict, Any, Optional

from utils.dataset import load_datasets
from utils.config import cfg

from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

def create_loaders() -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, test, and complete datasets.
    
    Returns:
        Tuple containing train, validation, test, and complete data loaders
    """
    # Load datasets
    train_dataset, validation_dataset, test_dataset, complete_dataset = load_datasets()
    
    # Configure DataLoader parameters
    loader_config: Dict[str, Any] = {
        'batch_size': cfg.train.batch_size
    }

    # Create loaders with appropriate settings
    train_loader = create_loader(
        dataset=train_dataset,
        shuffle=True,
        **loader_config
    )
    
    validation_loader = create_loader(
        dataset=validation_dataset,
        shuffle=False,
        **loader_config
    )
    
    test_loader = create_loader(
        dataset=test_dataset,
        shuffle=False,
        **loader_config
    )
    
    complete_loader = create_loader(
        dataset=complete_dataset,
        shuffle=False,
        **loader_config
    )

    return train_loader, validation_loader, test_loader, complete_loader

def create_loader(
    dataset: Dataset,
    shuffle: bool,
    **kwargs: Any
) -> DataLoader:
    """
    Create a DataLoader with consistent configuration.
    
    Args:
        dataset: Dataset to load
        shuffle: Whether to shuffle the data
        **kwargs: Additional arguments to pass to DataLoader
        
    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        **kwargs
    )
