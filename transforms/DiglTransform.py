import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

from rewiring.factory import RewireFactory

class DiglTransform(BaseTransform):
    """
    Transform that applies Diffusion Improves Graph Learning (DIGL) rewiring to input graphs.
    
    This transform implements the rewiring technique described in the paper
    "Diffusion Improves Graph Learning" which uses Personalized PageRank (PPR) to
    rewire graphs based on diffusion dynamics. It computes a PPR matrix for the 
    input graph and then either selects the top-k edges or clips edges based on a 
    threshold.
    
    The original edge structure is preserved in edge_index, while the rewired
    structure is added as rewiring_edge_index.
    """
    
    def __init__(self, alpha: float = 0.1, k: int = 128, eps: float = None) -> None:
        """
        Initialize the DiglTransform.
        
        Args:
            alpha: Teleport probability in PPR computation
            k: Number of neighbors to keep per node (if specified)
            eps: Threshold for edge clipping (if k is not specified)
        """
        super().__init__()
        self.alpha = alpha
        self.k = k
        self.eps = eps
        
        # Validate parameters
        if k is None and eps is None:
            self.eps = 0.01  # Default threshold

    def __call__(self, data: Data) -> Data:
        """
        Apply the DIGL rewiring transform to the input graph.
        
        This preserves the original edge_index and adds the rewired graph as
        rewiring_edge_index.
        
        Args:
            data: PyTorch Geometric Data object representing a graph
            
        Returns:
            The transformed Data object with DIGL rewiring added
        """
        # Get the number of nodes and original edge index
        num_nodes = data.num_nodes
        original_edge_index = data.edge_index
        
        # Create a custom DIGL strategy with the specified parameters
        digl_rewiring = RewireFactory.get_strategy('DIGL')
        digl_rewiring.alpha = self.alpha
        digl_rewiring.k = self.k
        digl_rewiring.eps = self.eps
        
        # Apply the rewiring strategy
        data.rewiring_edge_index = digl_rewiring.rewire(
            num_nodes=num_nodes, 
            original_edge_index=original_edge_index
        )
        
        return data 