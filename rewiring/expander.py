import torch
import numpy as np
from typing import Dict, Tuple, List, Deque, Optional, Any
from collections import deque

from primefac import primefac
from rewiring.base import RewireStrategy


class EGPRewiring(RewireStrategy):
    """
    Expander Graph Propagation (EGP) rewiring strategy.
    
    EGP uses Cayley graphs as expander structures to enhance message passing in
    graph neural networks. It truncates the Cayley graph to match the input graph size.
    """
    
    def __init__(self) -> None:
        """Initialize the EGP rewiring strategy with a Cayley graph generator."""
        self.cayley_generator = CayleyGraphGenerator()
    
    def rewire(self, num_nodes: int, original_edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply EGP rewiring to create an expander graph structure.
        
        Args:
            num_nodes: Number of nodes in the graph
            original_edge_index: Original edge index of the graph (not used)
            
        Returns:
            Edge index tensor for the truncated Cayley graph
        """
        cayley_n = self.cayley_generator.get_cayley_n(num_nodes)
        return self.cayley_generator.get_egp_edge_index(cayley_n, num_nodes)


class CGPRewiring(RewireStrategy):
    """
    Cayley Graph Propagation (CGP) rewiring strategy.
    
    CGP uses the full Cayley graph structure and marks extra nodes as virtual.
    It returns both the edge index and information about the number of nodes
    in the Cayley graph.
    """
    
    def __init__(self) -> None:
        """Initialize the CGP rewiring strategy with a Cayley graph generator."""
        self.cayley_generator = CayleyGraphGenerator()
    
    def rewire(self, num_nodes: int, original_edge_index: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, int]:
        """
        Apply CGP rewiring to create a full Cayley graph structure.
        
        Args:
            num_nodes: Number of nodes in the graph
            original_edge_index: Original edge index of the graph (not used)
            
        Returns:
            Tuple containing:
            - Edge index tensor for the full Cayley graph
            - Total number of nodes in the Cayley graph
        """
        cayley_n = self.cayley_generator.get_cayley_n(num_nodes)
        return self.cayley_generator.get_cgp_edge_index(cayley_n)


class CayleyGraphGenerator:
    """
    Generator class for Cayley graph structures.
    
    This class handles the creation and caching of Cayley graphs based on
    the special linear group SL(2,Z_n). It provides methods for computing
    graph sizes, generating complete Cayley graphs, and creating truncated
    versions for specific use cases.
    """
    
    def __init__(self) -> None:
        """
        Initialize the CayleyGraphGenerator.
        
        Sets up memory caches for storing precomputed Cayley graphs.
        """
        self.cayley_memory: Dict[int, torch.Tensor] = {}
        self.cayley_node_memory: Dict[int, torch.Tensor] = {}
    
    def get_cayley_n(self, num_nodes: int) -> int:
        """
        Determine the smallest Cayley graph parameter n such that the resulting 
        Cayley graph has at least num_nodes vertices.
        
        Args:
            num_nodes: Minimum number of nodes needed in the Cayley graph
            
        Returns:
            The smallest valid n parameter for the Cayley graph
        """
        n: int = 1
        while self.cayley_graph_size(n) < num_nodes:
            n += 1
        return n
    
    def cayley_graph_size(self, n: int) -> int:
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

    def get_egp_edge_index(self, cayley_n: int, num_nodes: int) -> torch.Tensor:
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
    
    def get_cgp_edge_index(self, cayley_n: int) -> Tuple[torch.Tensor, int]:
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
        cayley_num_nodes: int = self.cayley_graph_size(cayley_n)
        
        # Check if we've already computed this Cayley graph
        if cayley_n not in self.cayley_memory:
            edge_index: torch.Tensor = get_cayley_graph(cayley_n)
            self.cayley_memory[cayley_n] = edge_index

        edge_index: torch.Tensor = self.cayley_memory[cayley_n].clone()

        return edge_index, cayley_num_nodes
    
    def shuffle_edge_index_inplace(self, num_nodes: int, edge_index: torch.Tensor) -> None:
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
                queue.appendleft(tx)
                ind += 1
                
            ind_tx: int = nodes[tx_flat]
            
            # Add edges in both directions (undirected graph)
            senders.append(ind_x)
            receivers.append(ind_tx)
            senders.append(ind_tx)
            receivers.append(ind_x)

    # Convert edge lists to PyTorch tensors
    senders_tensor: torch.Tensor = torch.tensor(senders, dtype=torch.long)
    receivers_tensor: torch.Tensor = torch.tensor(receivers, dtype=torch.long)
    
    # Create edge_index in PyG format [2, num_edges]
    edge_index: torch.Tensor = torch.stack([senders_tensor, receivers_tensor], dim=0)
    
    return edge_index
