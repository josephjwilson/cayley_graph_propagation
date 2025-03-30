# DIGL pre-processing, from https://github.com/gasteigerjo/gdc.git

import numpy as np
import torch
from torch_geometric.data import Data
from typing import Optional, Tuple, Union

from rewiring.base import RewireStrategy


class DiglRewiring(RewireStrategy):
    """
    Diffusion Improves Graph Learning (DIGL) rewiring strategy.
    
    This rewiring technique uses Personalized PageRank (PPR) to rewire graphs based on
    diffusion dynamics. It computes a PPR matrix for the input graph and then either
    selects the top-k edges or clips edges based on a threshold, resulting in a graph 
    structure that better captures the diffusion properties.
    
    Implementation based on the paper "Diffusion Improves Graph Learning"
    (https://arxiv.org/abs/1911.05485).
    """
    
    def __init__(self, alpha: float = 0.1, k: int = 128, eps: float = None) -> None:
        """
        Initialize the DIGL rewiring strategy.
        
        Args:
            alpha: Teleport probability in PPR computation
            k: Number of neighbors to keep per node (if specified)
            eps: Threshold for edge clipping (if k is not specified)
        """
        self.alpha = alpha
        self.k = k
        self.eps = eps
        
        # Validate parameters
        if k is None and eps is None:
            self.eps = 0.01  # Default threshold
        
    def rewire(self, num_nodes: int, original_edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply DIGL rewiring to modify edge connectivity based on diffusion dynamics.
        
        Args:
            num_nodes: Number of nodes in the graph
            original_edge_index: Original edge index of the graph (required for DIGL)
            
        Returns:
            Edge index tensor for the rewired graph
            
        Raises:
            ValueError: If original_edge_index is not provided
        """
        if original_edge_index is None:
            raise ValueError("DIGL rewiring requires the original edge index")
            
        # Create a temporary Data object for the DIGL implementation
        base_data = Data(
            x=torch.zeros((num_nodes, 1)),  # Dummy features
            edge_index=original_edge_index,
            y=None  # No labels needed for rewiring
        )
        
        # Apply the DIGL rewiring
        if self.k is not None:
            rewired_edge_index = rewire(base_data, self.alpha, k=self.k)
        elif self.eps is not None:
            rewired_edge_index = rewire(base_data, self.alpha, eps=self.eps)
        else:
            raise ValueError("Either k or eps must be specified for DIGL rewiring")
            
        return rewired_edge_index


def get_adj_matrix(dataset) -> np.ndarray:
    num_nodes = dataset.x.shape[0]
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
    for i, j in zip(dataset.edge_index[0], dataset.edge_index[1]):
        adj_matrix[i, j] = 1.
    return adj_matrix

def get_ppr_matrix(
        adj_matrix: np.ndarray,
        alpha: float = 0.1) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H)

def get_top_k_matrix(A: np.ndarray, k: int = 128) -> np.ndarray:
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes)
    A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm

def get_clipped_matrix(A: np.ndarray, eps: float = 0.01) -> np.ndarray:
    num_nodes = A.shape[0]
    A[A < eps] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm

def rewire(base, alpha, k=None, eps=None):
    # generate adjacency matrix from sparse representation
    adj_matrix = get_adj_matrix(base)
    # obtain exact PPR matrix
    ppr_matrix = get_ppr_matrix(adj_matrix, alpha=alpha)

    if k != None:
            #print(f'Selecting top {k} edges per node.')
            ppr_matrix = get_top_k_matrix(ppr_matrix, k=k)
    elif eps != None:
            #print(f'Selecting edges with weight greater than {eps}.')
            ppr_matrix = get_clipped_matrix(ppr_matrix, eps=eps)
    else:
        raise ValueError

        # create PyG Data object
    edges_i = []
    edges_j = []
    edge_attr = []
    for i, row in enumerate(ppr_matrix):
        for j in np.where(row > 0)[0]:
            edges_i.append(i)
            edges_j.append(j)
            edge_attr.append(ppr_matrix[i, j])
    edge_index = [edges_i, edges_j]

    data = Data(
        x=base.x,
        edge_index=torch.LongTensor(edge_index),
        y=base.y
    )        
    return data.edge_index