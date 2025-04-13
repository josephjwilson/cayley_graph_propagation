# type: ignore

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

from rewiring.factory import RewireFactory

class GtrTransform(BaseTransform):
    """
    Transform that applies Graph Traversal Rewiring (GTR) to input graphs.
    
    This transform implements the rewiring technique described in the paper
    "Improving Graph Neural Networks with Graph Traversal Rewiring", which
    adds edges that most decrease the sum of effective resistances between nodes.
    It uses spectral graph theory and the graph Laplacian's pseudoinverse to
    identify optimal edge additions.
    
    The original edge structure is preserved in edge_index, while the rewired
    structure (with additional edges) is added as rewiring_edge_index.
    """
    
    def __init__(
        self, 
        num_edges: int = 10,
        try_gpu: bool = True
    ) -> None:
        """
        Initialize the GtrTransform.
        
        Args:
            num_edges: Number of edges to add to the graph
            try_gpu: Whether to use GPU acceleration if available
        """
        super().__init__()
        self.num_edges = num_edges
        self.try_gpu = try_gpu

    def __call__(self, data: Data) -> Data:
        """
        Apply the GTR rewiring transform to the input graph.
        
        This preserves the original edge_index and adds the rewired graph
        (with additional edges) as rewiring_edge_index.
        
        Args:
            data: PyTorch Geometric Data object representing a graph
            
        Returns:
            The transformed Data object with GTR rewiring added
        """
        # Get the number of nodes and original edge index
        num_nodes = data.num_nodes
        original_edge_index = data.edge_index
        
        # Create a custom GTR strategy with the specified parameters
        gtr_rewiring = RewireFactory.get_strategy('GTR')
        gtr_rewiring.num_edges = self.num_edges
        gtr_rewiring.try_gpu = self.try_gpu
        
        # Apply the rewiring strategy
        data.rewiring_edge_index = gtr_rewiring.rewire(
            num_nodes=num_nodes, 
            original_edge_index=original_edge_index
        )
        
        return data 