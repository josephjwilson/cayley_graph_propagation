# type: ignore

import torch

from utils.config import cfg
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

class PpaTransform(BaseTransform):
    """
    Transform to handle the OGB-PPA (Protein-Protein Association) dataset.
    
    This transform adds zero-initialized node features to the PPA dataset
    from the Open Graph Benchmark (OGB). The PPA dataset requires special
    handling for its node features during preprocessing.
    """
    
    def __init__(self) -> None:
        """
        Initialize the PpaTransform.
        """
        super().__init__()    

    def __call__(self, data: Data) -> Data:
        """
        Apply the transform to the graph data.
        
        For the OGB PPA dataset, this initializes node features as zero vectors
        with long data type, which is required for the PPA dataset's node embedding.
        
        Args:
            data: PyTorch Geometric Data object representing a graph
            
        Returns:
            The transformed Data object with node features initialized if needed
        """
        if cfg.dataset.name == "ogbg-ppa":
            data.x = torch.zeros(data.num_nodes, dtype=torch.long)
            
        return data
