# type: ignore

import torch

from utils.config import cfg
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

class TuTransform(BaseTransform):
    """
    Transform to handle TU datasets that do not have node features.
    
    This transform adds a simple feature vector of ones to nodes in graph datasets
    that don't have predefined node features, such as COLLAB, REDDIT-BINARY, 
    and IMDB-BINARY from the TU Dataset collection.
    """
    
    def __init__(self) -> None:
        """
        Initialize the TuTransform.
        """
        super().__init__()    

    def __call__(self, data: Data) -> Data:
        """
        Apply the transform to the graph data.
        
        For specific TU datasets that don't have node features (data.x is None),
        this adds a simple one-dimensional feature of ones to each node.
        
        Args:
            data: PyTorch Geometric Data object representing a graph
            
        Returns:
            The transformed Data object with node features added if needed
        """
        if cfg.dataset.name in ['COLLAB', 'REDDIT-BINARY', 'IMDB-BINARY'] and data.x is None:
            data.x = torch.ones((data.num_nodes, 1))

        return data
