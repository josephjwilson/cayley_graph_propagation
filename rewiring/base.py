import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

class RewireStrategy(ABC):
    """
    Abstract base class for graph rewiring strategies.
    
    Rewiring strategies modify the connectivity structure of a graph
    without changing the set of nodes. They can be used to enhance
    message passing in graph neural networks.
    
    All concrete rewiring implementations should inherit from this class
    and implement the rewire method.
    """
    
    @abstractmethod
    def rewire(self, num_nodes: int, original_edge_index: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply the rewiring strategy to create a new edge structure.
        
        Args:
            num_nodes: Number of nodes in the graph
            original_edge_index: Original edge index of the graph (optional)
            
        Returns:
            New edge index tensor or tuple of (edge index, additional information)
        """
        pass 