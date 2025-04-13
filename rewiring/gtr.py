# type: ignore
import torch
import torch_geometric
from torch_geometric.transforms import BaseTransform
from torch_geometric.data.datapipes import functional_transform
from typing import Optional, Tuple, Union

import networkx as nx
from functools import reduce

from rewiring.base import RewireStrategy

class GtrRewiring(RewireStrategy):
    """
    Graph Traversal Rewiring (GTR) strategy.
    
    This rewiring technique improves connectivity in graphs by adding edges that 
    most decrease the sum of effective resistances between nodes. It's based on 
    spectral graph theory and uses the graph Laplacian's pseudoinverse to identify
    optimal edge additions.
    
    Implementation based on the paper "Improving Graph Neural Networks with 
    Graph Traversal Rewiring" (https://arxiv.org/abs/2303.00839).
    """
    
    def __init__(
        self, 
        num_edges: int = 10,
        try_gpu: bool = True
    ) -> None:
        """
        Initialize the GTR rewiring strategy.
        
        Args:
            num_edges: Number of edges to add to the graph
            try_gpu: Whether to use GPU acceleration if available
        """
        self.num_edges = num_edges
        self.try_gpu = try_gpu
    
    def rewire(self, num_nodes: int, original_edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply GTR rewiring to add edges that improve the graph's connectivity.
        
        Args:
            num_nodes: Number of nodes in the graph
            original_edge_index: Original edge index of the graph (required for GTR)
            
        Returns:
            Edge index tensor for the rewired graph with added edges
            
        Raises:
            ValueError: If original_edge_index is not provided
        """
        if original_edge_index is None:
            raise ValueError("GTR rewiring requires the original edge index")
            
        # Create a temporary Data object for the GTR implementation
        data = type('Data', (), {})()
        data.x = torch.zeros((num_nodes, 1))  # Dummy features
        data.edge_index = original_edge_index
        data.num_nodes = num_nodes
        
        # Create GTR edge builder and compute new edges
        edge_builder = GTREdgeBuilder(data, self.try_gpu)
        new_edges = edge_builder.compute_edges(self.num_edges)
        
        # Combine original edges with new edges
        rewired_edge_index = torch.cat([original_edge_index, new_edges], dim=1)
        
        return rewired_edge_index


class GTREdgeBuilder:
    def __init__(
        self,
        data : torch_geometric.data.Data,
        try_gpu : bool = True
    ):
        if try_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.data = data
        self.graph = torch_geometric.utils.convert.to_networkx(data).to_undirected()
        self.laplacian = torch.tensor(nx.laplacian_matrix(self.graph).todense().astype("float")).to(self.device)
        self.pinv = torch.linalg.pinv(self.laplacian, hermitian=True)
        self.squared_pinv = self.pinv @ self.pinv
        self.edge_mask = self.compute_edge_mask()
        self.component_mask = self.compute_component_mask().to(self.device)

    def compute_edge_mask(
        self
    ) -> torch.Tensor:
        """ Return an n x n tensor where i,j entry is 1 iff {i,j} is not an edge in the graph or a self loop

        The non-zero of the Laplacian are the entries i,j where i==j or {i,j} is an edge in the graph.
        Thus, the logical_not of the laplacian is the edge mask.
        """
        return torch.logical_not(self.laplacian.bool())


    def compute_component_mask(
        self
    ) -> torch.Tensor:
        """ Return a n x n torch tensor where the i,j entry is 1 iff i and j are in the same connected component of `graph` """
        def single_component_mask(
            vertices_of_component : list
        ) -> torch.Tensor:
            """ Return a binary matrix where each i,j entry is 1 iff i and j are in `vertices_of_component` """
            row_mask = torch.zeros((self.data.num_nodes, self.data.num_nodes))
            col_mask = torch.zeros((self.data.num_nodes, self.data.num_nodes))
            row_mask[vertices_of_component,:] = 1
            col_mask[:,vertices_of_component] = 1
            return torch.logical_and(row_mask, col_mask)
        component_mask = reduce(
            torch.logical_or,
            [single_component_mask(list(component)) for component in nx.connected_components(self.graph)]
        )
        return component_mask

    def update_edge_mask(
        self,
        s : int,
        t : int
    ):
        """ Update the edge mask after adding the edge {s,t} """
        self.edge_mask[s,t] = 0
        self.edge_mask[t,s] = 0

    def update_laplacian(
        self,
        s : int,
        t : int
    ):
        """ Update the Laplacian after adding the edge {s,t} """
        self.laplacian[s,s] += 1; self.laplacian[s,t] -= 1
        self.laplacian[t,s] -= 1; self.laplacian[t,t] += 1

    def update_pseudoinvere(
        self,
        s : int,
        t : int
    ):
        """ Update the pseudoinverse of the Laplacian with Woodbury's formula after adding the edge {s,t} """
        v = self.pinv[:,s] - self.pinv[:,t]
        effective_resistance = v[s] - v[t]
        self.pinv = self.pinv - (1/(1+effective_resistance))*torch.outer(v, v)

    def update_squared_pseudoinverse(
        self,
        s : int,
        t : int
    ):
        """ Update the squared pseudoinverse of the Laplacian with Woodbury's formula after adding the edge {s,t} """
        x = torch.zeros(self.data.num_nodes).to(self.device)
        x[s], x[t] = 1, -1
        y = self.laplacian[:,s] - self.laplacian[:,t]
        U = torch.column_stack([x, y+x])
        V = torch.stack([y+x, x])
        left = self.squared_pinv @ U
        center = torch.inverse(torch.eye(2).to(self.device) + V@self.squared_pinv@U)
        right = V @ self.squared_pinv
        self.squared_pinv = self.squared_pinv - left@center@right

    def compute_edges(
        self,
        num_edges : int,
    ) -> torch.Tensor:
        """ Calculate edges to add to the graph using the GTR heuristic. """
        ret_edges = torch.zeros((2,0), dtype=torch.long)
        for _ in range(num_edges):
            # The entries resistance_matrix[s,t] and biharmonic_matrix[s,t]
            # are the effective resistance and biharmonic distance between s and t.
            pinv_diagonal = torch.diagonal(self.pinv)
            resistance_matrix = pinv_diagonal.unsqueeze(0) + pinv_diagonal.unsqueeze(1) - 2*self.pinv
            squared_pinv_diagonal = torch.diagonal(self.squared_pinv)
            biharmonic_matrix = squared_pinv_diagonal.unsqueeze(0) + squared_pinv_diagonal.unsqueeze(1) - 2*self.squared_pinv
            # diff_matrix[s,t] stores the change in total resistance when the edge {s,t} is added to the graph
            diff_matrix = (biharmonic_matrix / (1 + resistance_matrix))
            # We only want to add an edge not already in the graph and not between connected components.
            # Multiplying by edge_mask sets the value of all self-loops
            # and edges already in the graph to 0.
            # Multiplying by component_mask sets the value
            # of all edges between connected components to 0.
            masked_diff_matrix = diff_matrix * self.edge_mask * self.component_mask
            # Find the endpoints of the edge that most decrease the total resistance.
            # torch.argmax returns the index of max in flattened coordinates, hence the divmod.
            s, t = divmod(torch.argmax(masked_diff_matrix).cpu().item(), self.data.num_nodes)
            # Add the edge {s,t} to the return array
            new_edges = torch.Tensor([[s, t], [t, s]]).long()
            ret_edges = torch.cat([ret_edges, new_edges], 1)
            # Update the matrices for the next iteration
            self.update_pseudoinvere(s, t)
            self.update_squared_pseudoinverse(s, t)
            self.update_laplacian(s, t)
            self.update_edge_mask(s, t)
        return ret_edges


@functional_transform('add_gtr_edges')
class AddGTREdges(BaseTransform):
    def __init__(
        self,
        num_edges : int,
        try_gpu : bool = True
    ):
        print(
            "We do not recommend using AddGTREdges.",
            "AddGTREdges computes additional edges each time a graph is loaded,",
            "which is computationally expensive.",
            "We recommend using PrecomputeGTREdges and AddPrecomputedGTREdges instead."
        )
        self.num_edges = num_edges
        self.try_gpu = try_gpu

    def __call__(
        self,
        data : torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        # add edge types
        if hasattr(data, "edge_type"):
            # if data already has edge types, label new edges with next unused int
            new_edge_type = torch.max(data.edge_type) + 1
            data.edge_type = torch.cat([
                data.edge_type,
                torch.full((2*self.num_edges,), new_edge_type, dtype=torch.long)
            ])
        else:
            # if data does not already have edge types, label original edges 0 and new edges 1
            num_original_edges = data.edge_index.shape[1]
            data.edge_type = torch.cat([
                torch.zeros(num_original_edges, dtype=torch.long),
                torch.ones(2*self.num_edges, dtype=torch.long)
            ])
        # add edges
        new_edges =  GTREdgeBuilder(data, self.try_gpu).compute_edges(self.num_edges)
        data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
        return data


@functional_transform('add_precomputed_gtr_edges')
class AddPrecomputedGTREdges(BaseTransform):
    """ AddPrecomputedGTREdges adds a specified number of precomputed edges to a Data object.

    The AddPrecomputedGTREdges transform adds a specified number of precomputed edges to a
    torch_geometric.data.Data object. AddPrecomputedGTREdges must be used in conjuction with
    the PrecomputeGTREdges.
    """
    def __init__(
        self,
        num_edges : int
    ):
        self.num_edges = num_edges

    def __call__(
        self,
        data : torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        if not hasattr(data, "precomputed_gtr_edges"):
            raise AttributeError("Data does not have precomputed edges. Edges must be computed with the pre_transform PrecomputeGTREdges.")
        if data.precomputed_gtr_edges.shape[1] < 2*self.num_edges:
            raise ValueError("Too few edges have been precomputed.")
        # add edge types
        if hasattr(data, "edge_type"):
            # if data already has edge types, label new edges with next unused int
            new_edge_type = torch.max(data.edge_type) + 1
            data.edge_type = torch.cat([
                data.edge_type,
                torch.full((2*self.num_edges,), new_edge_type, dtype=torch.long)
            ])
        else:
            # if data does not already have edge types, label original edges 0 and new edges 1
            num_original_edges = data.edge_index.shape[1]
            data.edge_type = torch.cat([
                torch.zeros(num_original_edges, dtype=torch.long),
                torch.ones(2*self.num_edges, dtype=torch.long)
            ])
        # add new edges
        new_edges = data.precomputed_gtr_edges[:,:2*self.num_edges]
        data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
        return data

@functional_transform('precompute_gtr_edges')
class PrecomputeGTREdges(BaseTransform):
    """ PrecomputeGTREdges precomputes a specified number of edges with the GTR heuristic.

    The PrecomputeGTREdges transform precomputes a specified number of edges to add with the GTR heuristic and stores
    them as the attribute `precomputed_gtr_edges` in a torch_geometric Data object; however, PrecomputeGTREdges
    does not actually add the edges to the graph. To do this, PrecomputeGTREdges must be used in conjuction with
    the AddGTREdges transform.
    """
    def __init__(
        self,
        num_edges : int,
        try_gpu : bool = True
    ):
        self.num_edges = num_edges
        self.try_gpu = try_gpu

    def __call__(
        self,
        data : torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        data.precomputed_gtr_edges = GTREdgeBuilder(data, self.try_gpu).compute_edges(self.num_edges)
        return data