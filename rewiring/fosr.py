# Code courtesy of: https://github.com/kedar2/FoSR
from numba import jit
import numpy as np
import torch
from math import inf
from typing import Optional, Tuple, Union

from rewiring.base import RewireStrategy


class FosrRewiring(RewireStrategy):
    """
    First-Order Spectral Rewiring (FoSR) strategy.
    
    This rewiring technique is designed to improve the spectral properties of graphs
    by optimizing edge placement to increase the spectral gap of the normalized 
    Laplacian. It iteratively adds edges that minimize the product of the Fiedler
    vector entries.
    
    Implementation is based on the paper "Improving Graph Neural Network Expressivity 
    via Subgraph Isomorphism Counting" (https://arxiv.org/abs/2006.09252).
    """
    
    def __init__(self, num_iterations: int = 50, initial_power_iters: int = 5) -> None:
        """
        Initialize the FoSR rewiring strategy.
        
        Args:
            num_iterations: Number of edge addition iterations
            initial_power_iters: Number of power iterations for eigenvector approximation
        """
        self.num_iterations = num_iterations
        self.initial_power_iters = initial_power_iters
    
    def rewire(self, num_nodes: int, original_edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply FoSR rewiring to optimize edge placement based on spectral properties.
        
        Args:
            num_nodes: Number of nodes in the graph
            original_edge_index: Original edge index of the graph (required for FoSR)
            
        Returns:
            Edge index tensor for the rewired graph
        
        Raises:
            ValueError: If original_edge_index is not provided
        """
        if original_edge_index is None:
            raise ValueError("FoSR rewiring requires the original edge index")
        
        # Convert PyTorch edge_index to NumPy for FOSR implementation
        edge_np = original_edge_index.cpu().numpy()
        
        # Initialize placeholder edge types (for compatibility with the original code)
        edge_type = np.zeros(edge_np.shape[1], dtype=np.int64)
        
        # Apply the original FOSR edge rewiring
        rewired_edge_index, _, _ = edge_rewire(
            edge_np, 
            edge_type=edge_type,
            num_iterations=self.num_iterations,
            initial_power_iters=self.initial_power_iters
        )
        
        # Convert back to PyTorch tensor
        return torch.tensor(rewired_edge_index, dtype=torch.long)


@jit(nopython=True)
def choose_edge_to_add(x, edge_index, degrees):
	# chooses edge (u, v) to add which minimizes y[u]*y[v]
	n = x.size
	m = edge_index.shape[1]
	y = x / ((degrees + 1) ** 0.5)
	products = np.outer(y, y)
	for i in range(m):
		u = edge_index[0, i]
		v = edge_index[1, i]
		products[u, v] = inf
	for i in range(n):
		products[i, i] = inf
	smallest_product = np.argmin(products)
	return (smallest_product % n, smallest_product // n)

@jit(nopython=True)
def compute_degrees(edge_index, num_nodes=None):
	# returns array of degrees of all nodes
	if num_nodes is None:
		num_nodes = np.max(edge_index) + 1
	degrees = np.zeros(num_nodes)
	m = edge_index.shape[1]
	for i in range(m):
		degrees[edge_index[0, i]] += 1
	return degrees

@jit(nopython=True)
def add_edge(edge_index, u, v):
	new_edge = np.array([[u, v],[v, u]])
	return np.concatenate((edge_index, new_edge), axis=1)

@jit(nopython=True)
def adj_matrix_multiply(edge_index, x):
	# given an edge_index, computes Ax, where A is the corresponding adjacency matrix
	n = x.size
	y = np.zeros(n)
	m = edge_index.shape[1]
	for i in range(m):
		u = edge_index[0, i]
		v = edge_index[1, i]
		y[u] += x[v]
	return y

@jit(nopython=True)
def compute_spectral_gap(edge_index, x):
	m = edge_index.shape[1]
	n = np.max(edge_index) + 1
	degrees = compute_degrees(edge_index, num_nodes=n)
	y = adj_matrix_multiply(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
	for i in range(n):
		if x[i] > 1e-9:
			return 1 - y[i]/x[i]
	return 0.

@jit(nopython=True)
def _edge_rewire(edge_index, edge_type, x=None, num_iterations=50, initial_power_iters=50):
	m = edge_index.shape[1]
	n = np.max(edge_index) + 1
	if x is None:
		x = 2 * np.random.random(n) - 1
	degrees = compute_degrees(edge_index, num_nodes=n)
	for i in range(initial_power_iters):
		x = x - x.dot(degrees ** 0.5) * (degrees ** 0.5)/sum(degrees)
		y = x + adj_matrix_multiply(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
		x = y / np.linalg.norm(y)
	for I in range(num_iterations):
		i, j = choose_edge_to_add(x, edge_index, degrees=degrees)
		edge_index = add_edge(edge_index, i, j)
		degrees[i] += 1
		degrees[j] += 1
		edge_type = np.append(edge_type, 1)
		edge_type = np.append(edge_type, 1)
		x = x - x.dot(degrees ** 0.5) * (degrees ** 0.5)/sum(degrees)
		y = x + adj_matrix_multiply(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
		x = y / np.linalg.norm(y)
	return edge_index, edge_type, x

def edge_rewire(edge_index, x=None, edge_type=None, num_iterations=50, initial_power_iters=5):
	m = edge_index.shape[1]
	n = np.max(edge_index) + 1
	if x is None:
		x = 2 * np.random.random(n) - 1
	if edge_type is None:
		edge_type = np.zeros(m, dtype=np.int64)
	return _edge_rewire(edge_index, edge_type=edge_type, x=x, num_iterations=num_iterations, initial_power_iters=initial_power_iters)