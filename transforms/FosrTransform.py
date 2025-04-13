# type: ignore

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

from rewiring.factory import RewireFactory

class FosrTransform(BaseTransform):
    """
    Transform that applies First-Order Spectral Rewiring (FoSR) to input graphs.
    
    This transform implements the rewiring technique described in the paper
    "Improving Graph Neural Network Expressivity via Subgraph Isomorphism Counting"
    which optimizes edge placement to increase the spectral gap of the normalized 
    Laplacian. It iteratively adds edges that minimize the product of the Fiedler
    vector entries.
    
    The original edge structure is preserved in edge_index, while the rewired
    structure is added as rewiring_edge_index.
    """
    
    def __init__(self, num_iterations: int = 50, initial_power_iters: int = 5) -> None:
        """
        Initialize the FosrTransform.
        
        Args:
            num_iterations: Number of edge addition iterations
            initial_power_iters: Number of power iterations for eigenvector approximation
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.initial_power_iters = initial_power_iters

    def __call__(self, data: Data) -> Data:
        """
        Apply the FoSR rewiring transform to the input graph.
        
        This preserves the original edge_index and adds the rewired graph as
        rewiring_edge_index.
        
        Args:
            data: PyTorch Geometric Data object representing a graph
            
        Returns:
            The transformed Data object with FoSR rewiring added
        """
        # Get the number of nodes and original edge index
        num_nodes = data.num_nodes
        original_edge_index = data.edge_index
        
        # Create a custom FoSR strategy with the specified parameters
        fosr_rewiring = RewireFactory.get_strategy('FoSR')
        fosr_rewiring.num_iterations = self.num_iterations
        fosr_rewiring.initial_power_iters = self.initial_power_iters
        
        # Apply the rewiring strategy
        data.rewiring_edge_index = fosr_rewiring.rewire(
            num_nodes=num_nodes, 
            original_edge_index=original_edge_index
        )
        
        return data
