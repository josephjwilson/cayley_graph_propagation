import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

from rewiring.factory import RewireFactory

class FullyAdjacentTransform(BaseTransform):
    """
    Transform that adds a fully connected edge structure to a graph.
    
    This transform creates edges between all pairs of nodes in the graph,
    effectively making it a complete graph. Self-loops are removed to
    maintain standard graph neural network conventions. The original
    edge structure is preserved, and the new fully connected edge structure
    is added as a separate attribute.
    """
    
    def __init__(self) -> None:
        """
        Initialize the FullyAdjacentTransform.
        """
        super().__init__()

    def __call__(self, data: Data) -> Data:
        """
        Apply the transform to the graph data.
        
        Creates a fully connected edge structure (every node connected to every other node)
        and stores it in data.rewiring_edge_index without self-loops. The original
        edge_index attribute is preserved.
        
        Args:
            data: PyTorch Geometric Data object representing a graph
            
        Returns:
            The transformed Data object with the fully connected edge structure
            added as rewiring_edge_index
        """
        # Get fully connected edge structure from the rewiring strategy
        rewiring = RewireFactory.get_strategy('FullyAdjacent')
        data.rewiring_edge_index = rewiring.rewire(data.num_nodes)
        
        return data
