import torch
from torch_geometric.utils import remove_self_loops
from typing import Optional

from rewiring.base import RewireStrategy

class FullyAdjacentRewiring(RewireStrategy):
    """
    Rewiring strategy that creates a fully connected graph.
    
    This strategy generates edges between all pairs of nodes, creating
    a complete graph structure without self-loops.
    """
    
    def rewire(self, num_nodes: int, original_edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply fully connected rewiring to create a complete graph.
        
        Args:
            num_nodes: Number of nodes in the graph
            original_edge_index: Original edge index of the graph (not used)
            
        Returns:
            Edge index tensor for the fully connected graph
        """
        return get_fully_adjacent_edge_index(num_nodes)


def get_fully_adjacent_edge_index(num_nodes: int) -> torch.Tensor:
    """
    Creates a fully connected edge structure (complete graph).
    
    This function generates edges between all pairs of nodes, creating
    a complete graph structure without self-loops.
    
    Args:
        num_nodes: Number of nodes in the graph
        
    Returns:
        Edge index tensor for the fully connected graph in PyTorch Geometric format,
        with self-loops removed
    """
    all_nodes: torch.Tensor = torch.arange(0, num_nodes)
    fully_edge_index: torch.Tensor = torch.cartesian_prod(all_nodes, all_nodes).T
    
    # Remove self-loops from the fully connected edge structure
    fully_edge_index_no_sl, _ = remove_self_loops(fully_edge_index)

    return fully_edge_index_no_sl 