import torch
import numpy as np

from utils.config import cfg
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from typing import Dict, Tuple, List

from rewiring.factory import RewireFactory

class ExpanderTransform(BaseTransform):
    """
    Transform that applies expander graph structures to input graphs.
    
    This transform implements both Expander Graph Propagation (EGP) and
    Cayley Graph Propagation (CGP) methods. Both techniques use Cayley graphs
    as expander structures to enhance message passing in graph neural networks.
    
    EGP truncates the Cayley graph to match the input graph size, while CGP
    keeps the complete Cayley graph structure and marks extra nodes as virtual.
    Both approaches store the expander structure in rewiring_edge_index.
    
    The transform includes caching mechanisms to avoid recomputing Cayley 
    graphs for the same parameters multiple times.
    """
    
    def __init__(self) -> None:
        """
        Initialize the ExpanderTransform.
        
        Sets up the rewiring strategies from the factory.
        """
        super().__init__()

    def __call__(self, data: Data) -> Data:
        """
        Apply the expander graph transform to the input graph.
        
        This method handles feature initialization for specific datasets and
        applies either EGP or CGP based on the configuration. For EGP, it truncates
        the Cayley graph to match the number of nodes. For CGP, it keeps the full
        Cayley graph and marks additional nodes as virtual.
        
        Args:
            data: PyTorch Geometric Data object representing a graph
            
        Returns:
            The transformed Data object with expander structure added
            
        Raises:
            ValueError: If an unknown transform type is specified
        """
        num_nodes: int = data.num_nodes

        # Handle OGB-PPA dataset initialization
        if cfg.dataset.name == "ogbg-ppa":
            data.x = torch.zeros(data.num_nodes, dtype=torch.long)

        # Handle feature initialization for TU datasets
        if cfg.dataset.name in ['COLLAB', 'REDDIT-BINARY', 'IMDB-BINARY'] and data.x is None:
            # Need to augment X
            data.x = torch.ones((num_nodes, 1))

        # Skip transform if no transformation is specified
        if cfg.transform.name is None:
            return data

        # Apply Expander Graph Propagation (EGP)
        if cfg.transform.name == 'EGP':
            rewiring = RewireFactory.get_strategy('EGP')
            data.rewiring_edge_index = rewiring.rewire(num_nodes)
            return data
            
        # Apply Cayley Graph Propagation (CGP)
        elif cfg.transform.name == 'CGP':
            rewiring = RewireFactory.get_strategy('CGP')
            data.rewiring_edge_index, cayley_num_nodes = rewiring.rewire(num_nodes)

            # Get the number of virtual nodes needed
            virtual_num_nodes: int = cayley_num_nodes - num_nodes

            # Create a boolean mask to indicate if the node is a virtual node
            data.virtual_node_mask = torch.cat((
                torch.zeros(num_nodes, dtype=torch.bool), 
                torch.ones(virtual_num_nodes, dtype=torch.bool)
            ), axis=0)

            # Update the input features to have the zero-node embeddings for the virtual nodes
            data.num_nodes = cayley_num_nodes
            data.cayley_num_nodes = cayley_num_nodes
            data.x = torch.cat((
                data.x, 
                torch.zeros((virtual_num_nodes, data.x.shape[1]), dtype=data.x.dtype)
            ), axis=0)

            return data
        else:
            raise ValueError(f'Expander transform does not exist: {cfg.transform.name}')
