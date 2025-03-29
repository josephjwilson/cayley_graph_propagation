import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data

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
        all_nodes: torch.Tensor = torch.arange(0, data.num_nodes)
        fully_edge_index: torch.Tensor = torch.cartesian_prod(all_nodes, all_nodes).T
        
        # Remove self-loops from the fully connected edge structure
        fully_edge_index_no_sl, _ = remove_self_loops(fully_edge_index)

        # Store the fully connected edge structure as rewiring_edge_index
        data.rewiring_edge_index = fully_edge_index_no_sl

        return data
