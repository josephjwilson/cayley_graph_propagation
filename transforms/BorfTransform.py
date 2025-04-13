# type: ignore

from typing import Optional
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

from rewiring.factory import RewireFactory

class BorfTransform(BaseTransform):
    """
    Transform that applies Balanced Optimal Ricci Flow (BORF) rewiring to input graphs.
    
    This transform implements the rewiring technique described in the paper
    "Understanding Over-squashing and Bottlenecks on Graphs via Curvature", which
    uses Ollivier-Ricci curvature to identify negatively curved edges and add new
    shortcut edges to improve the flow of information through the graph. It also
    optionally removes edges with high curvature.
    
    The original edge structure is preserved in edge_index, while the rewired
    structure is added as rewiring_edge_index.
    """
    
    def __init__(
        self, 
        loops: int = 10,
        remove_edges: bool = True,
        removal_bound: float = 0.5,
        tau: float = 1,
        is_undirected: bool = False,
        batch_add: int = 4,
        batch_remove: int = 2,
        algorithm: str = 'borf3',
        device = None,
        save_dir: str = 'rewired_graphs',
        dataset_name: Optional[str] = None,
        graph_index: int = 0,
        debug: bool = False
    ) -> None:
        """
        Initialize the BorfTransform.
        
        Args:
            loops: Number of rewiring iterations
            remove_edges: Whether to also remove high-curvature edges
            removal_bound: Minimum curvature for edge removal
            tau: Temperature parameter for softmax sampling of edges
            is_undirected: Whether the graph is undirected
            batch_add: Number of edges to add in each iteration
            batch_remove: Number of edges to remove in each iteration
            algorithm: Which BORF algorithm to use ('borf2' or 'borf3')
            device: Device to use for computations (CPU or CUDA)
            save_dir: Directory to save rewired graphs (for borf3)
            dataset_name: Name of the dataset (for borf3)
            graph_index: Index of the graph in the dataset (for borf3)
            debug: Whether to print debug information (for borf3)
        """
        super().__init__()
        self.loops = loops
        self.remove_edges = remove_edges
        self.removal_bound = removal_bound
        self.tau = tau
        self.is_undirected = is_undirected
        self.batch_add = batch_add
        self.batch_remove = batch_remove
        self.algorithm = algorithm
        self.device = device
        self.save_dir = save_dir
        self.dataset_name = dataset_name
        self.graph_index = graph_index
        self.debug = debug

    def __call__(self, data: Data) -> Data:
        """
        Apply the BORF rewiring transform to the input graph.
        
        This preserves the original edge_index and adds the rewired graph as
        rewiring_edge_index.
        
        Args:
            data: PyTorch Geometric Data object representing a graph
            
        Returns:
            The transformed Data object with BORF rewiring added
        """
        # Get the number of nodes and original edge index
        num_nodes = data.num_nodes
        original_edge_index = data.edge_index
        
        # Create a custom BORF strategy with the specified parameters
        borf_rewiring = RewireFactory.get_strategy('BORF')

        if borf_rewiring is None:
            raise ValueError("Failed to create BORF rewiring strategy")

        borf_rewiring.loops = self.loops
        borf_rewiring.remove_edges = self.remove_edges
        borf_rewiring.removal_bound = self.removal_bound
        borf_rewiring.tau = self.tau
        borf_rewiring.is_undirected = self.is_undirected
        borf_rewiring.batch_add = self.batch_add
        borf_rewiring.batch_remove = self.batch_remove
        borf_rewiring.algorithm = self.algorithm
        borf_rewiring.device = self.device
        borf_rewiring.save_dir = self.save_dir
        borf_rewiring.dataset_name = self.dataset_name
        borf_rewiring.graph_index = self.graph_index
        borf_rewiring.debug = self.debug
        
        # Apply the rewiring strategy
        data.rewiring_edge_index = borf_rewiring.rewire(
            num_nodes=num_nodes, 
            original_edge_index=original_edge_index
        )
        
        return data 