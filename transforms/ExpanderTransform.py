import torch
import numpy as np

from utils.config import cfg
from primefac import primefac
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from typing import Dict, Tuple, List, Deque
from collections import deque

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
        
        Sets up memory caches for storing precomputed Cayley graphs.
        """
        super().__init__()

        self.cayley_memory: Dict[int, torch.Tensor] = {}
        self.cayley_node_memory: Dict[int, torch.Tensor] = {}

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

        # Determine the Cayley graph parameter based on input graph size
        cayley_n: int = self._get_cayley_n(num_nodes)

        # Apply Expander Graph Propagation (EGP)
        if cfg.transform.name == 'EGP':
            data.rewiring_edge_index = self._get_egp_edge_index(cayley_n, num_nodes)
            return data
            
        # Apply Cayley Graph Propagation (CGP)
        elif cfg.transform.name == 'CGP':
            data.rewiring_edge_index, cayley_num_nodes = self._get_cgp_edge_index(cayley_n)

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

    def _get_cayley_n(self, num_nodes: int) -> int:
        """
        Determine the smallest Cayley graph parameter n such that the resulting 
        Cayley graph has at least num_nodes vertices.
        
        Args:
            num_nodes: Minimum number of nodes needed in the Cayley graph
            
        Returns:
            The smallest valid n parameter for the Cayley graph
        """
        n: int = 1
        while self._cayley_graph_size(n) < num_nodes:
            n += 1
        return n
    
    def _cayley_graph_size(self, n: int) -> int:
        """
        Calculate the size (number of nodes) of the Cayley graph for parameter n.
        
        The size is computed using number theory based on the special linear group SL(2,Z_n).
        
        Args:
            n: The parameter for the Cayley graph
            
        Returns:
            The number of nodes in the Cayley graph
        """
        n = int(n)
        return round(n*n*n*np.prod([1 - 1.0/(p * p) for p in list(set(primefac(n)))]))    

    def _get_egp_edge_index(self, cayley_n: int, num_nodes: int) -> torch.Tensor:
        """
        Get the edge index for Expander Graph Propagation (EGP).
        
        This method retrieves or creates a Cayley graph for parameter cayley_n,
        then truncates it to include only the first num_nodes nodes and their
        interconnections.
        
        Args:
            cayley_n: The parameter for the Cayley graph
            num_nodes: Number of nodes in the original graph
            
        Returns:
            Edge index tensor for the truncated Cayley graph
        """
        # Check if we've already computed this Cayley graph
        if cayley_n not in self.cayley_memory:
            self.cayley_memory[cayley_n] = get_cayley_graph(cayley_n)
        
        cayley_graph_edge_index: torch.Tensor = self.cayley_memory[cayley_n].clone()

        # Check if we've already truncated this graph for this number of nodes
        if num_nodes not in self.cayley_node_memory:
            # Keep only edges where both endpoints are within the first num_nodes
            truncated_edge_index: torch.Tensor = cayley_graph_edge_index[:, torch.logical_and(
                cayley_graph_edge_index[0] < num_nodes, 
                cayley_graph_edge_index[1] < num_nodes
            )]
            self.cayley_node_memory[num_nodes] = truncated_edge_index
        
        edge_index: torch.Tensor = self.cayley_node_memory[num_nodes].clone()

        return edge_index
    
    def _get_cgp_edge_index(self, cayley_n: int) -> Tuple[torch.Tensor, int]:
        """
        Get the edge index for Cayley Graph Propagation (CGP).
        
        This method retrieves or creates a Cayley graph for parameter cayley_n,
        and returns it without truncation along with the total number of nodes.
        
        Args:
            cayley_n: The parameter for the Cayley graph
            
        Returns:
            Tuple containing:
            - Edge index tensor for the complete Cayley graph
            - Number of nodes in the Cayley graph
        """
        cayley_num_nodes: int = self._cayley_graph_size(cayley_n)
        
        # Check if we've already computed this Cayley graph
        if cayley_n not in self.cayley_memory:
            edge_index: torch.Tensor = get_cayley_graph(cayley_n)
            self.cayley_memory[cayley_n] = edge_index

        edge_index: torch.Tensor = self.cayley_memory[cayley_n].clone()

        return edge_index, cayley_num_nodes
    
    def _shuffle_edge_index_inplace(self, num_nodes: int, edge_index: torch.Tensor) -> None:
        """
        Shuffle the edge index in-place using a random permutation of nodes.
        
        This method is useful for creating random variations of the graph structure.
        
        Args:
            num_nodes: Number of nodes in the graph
            edge_index: Edge index tensor to shuffle (modified in-place)
        """
        perm: np.ndarray = np.random.permutation(num_nodes)
        for i in range(edge_index.shape[1]):
            edge_index[0][i] = perm[edge_index[0][i]]
            edge_index[1][i] = perm[edge_index[1][i]]

def get_cayley_graph(n: int) -> torch.Tensor:
    """
    Generate the edge index of the Cayley graph Cay(SL(2, Z_n); S_n).
    
    This function creates a Cayley graph using the special linear group SL(2, Z_n)
    with a specific generating set. The graph construction uses a breadth-first
    approach to enumerate all nodes and edges.
    
    Args:
        n: Parameter for the Cayley graph (defines the ring Z_n)
        
    Returns:
        Edge index tensor for the Cayley graph in PyTorch Geometric format
    """
    generators: np.ndarray = np.array([
        [[1, 1], [0, 1]],
        [[1, n-1], [0, 1]],
        [[1, 0], [1, 1]],
        [[1, 0], [n-1, 1]]])
    ind: int = 1

    queue: Deque[np.ndarray] = deque([np.array([[1, 0], [0, 1]])])
    nodes: Dict[Tuple[int, int, int, int], int] = {(1, 0, 0, 1): 0}

    senders: List[int] = []
    receivers: List[int] = []

    while queue:
        x: np.ndarray = queue.pop()
        x_flat: Tuple[int, int, int, int] = (x[0][0], x[0][1], x[1][0], x[1][1])
        assert x_flat in nodes
        ind_x: int = nodes[x_flat]
        
        # Apply each generator to the current element
        for i in range(4):
            tx: np.ndarray = np.matmul(x, generators[i])
            tx = np.mod(tx, n)
            tx_flat: Tuple[int, int, int, int] = (tx[0][0], tx[0][1], tx[1][0], tx[1][1])
            
            # If we've found a new group element, add it to our nodes
            if tx_flat not in nodes:
                nodes[tx_flat] = ind
                ind += 1
                queue.append(tx)
                
            ind_tx: int = nodes[tx_flat]

            # Add the edge to our edge list
            senders.append(ind_x)
            receivers.append(ind_tx)
            
    return torch.tensor([senders, receivers])
