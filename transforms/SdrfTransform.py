import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

from rewiring.factory import RewireFactory

class SdrfTransform(BaseTransform):
    """
    Transform that applies Stochastic Discrete Ricci Flow (SDRF) rewiring to input graphs.
    
    This transform implements the rewiring technique described in the paper
    "Understanding Oversquashing and Bottlenecks on Graphs via Curvature" which
    addresses oversquashing in graph neural networks by optimizing the graph 
    structure using discrete Ricci curvature. It iteratively adds edges based on 
    the Balanced Forman curvature to improve message flow through the graph.
    
    The original edge structure is preserved in edge_index, while the rewired
    structure is added as rewiring_edge_index.
    """
    
    def __init__(
        self, 
        loops: int = 10, 
        remove_edges: bool = True, 
        removal_bound: float = 0.5, 
        tau: float = 1, 
        is_undirected: bool = False
    ) -> None:
        """
        Initialize the SdrfTransform.
        
        Args:
            loops: Number of rewiring iterations
            remove_edges: Whether to also remove high-curvature edges
            removal_bound: Minimum curvature for edge removal
            tau: Temperature parameter for softmax sampling of edges
            is_undirected: Whether the graph is undirected
        """
        super().__init__()
        self.loops = loops
        self.remove_edges = remove_edges
        self.removal_bound = removal_bound
        self.tau = tau
        self.is_undirected = is_undirected

    def __call__(self, data: Data) -> Data:
        """
        Apply the SDRF rewiring transform to the input graph.
        
        This preserves the original edge_index and adds the rewired graph as
        rewiring_edge_index.
        
        Args:
            data: PyTorch Geometric Data object representing a graph
            
        Returns:
            The transformed Data object with SDRF rewiring added
        """
        # Get the number of nodes and original edge index
        num_nodes = data.num_nodes
        original_edge_index = data.edge_index
        
        # Create a custom SDRF strategy with the specified parameters
        sdrf_rewiring = RewireFactory.get_strategy('SDRF')
        sdrf_rewiring.loops = self.loops
        sdrf_rewiring.remove_edges = self.remove_edges
        sdrf_rewiring.removal_bound = self.removal_bound
        sdrf_rewiring.tau = self.tau
        sdrf_rewiring.is_undirected = self.is_undirected
        
        # Apply the rewiring strategy
        data.rewiring_edge_index = sdrf_rewiring.rewire(
            num_nodes=num_nodes, 
            original_edge_index=original_edge_index
        )
        
        return data 