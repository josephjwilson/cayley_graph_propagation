import torch
import numpy as np

from utils.config import cfg
from primefac import primefac
from torch_geometric.transforms import BaseTransform
from typing import Dict
from collections import deque

# This class also handle some of the TUDataset Transform
class ExpanderTransform(BaseTransform):
    def __init__(self):
        super(ExpanderTransform).__init__()

        self.cayley_memory: Dict[int, torch.Tensor] = {}
        self.cayley_node_memory: Dict[int, torch.Tensor] = {}

    def __call__(self, data):
        num_nodes = data.num_nodes

        if cfg.dataset.name == "ogbg-ppa":
            data.x = torch.zeros(data.num_nodes, dtype=torch.long)

        # For certain TUDataset(s) the graph structure needs to be augmented, as they do not have node features
        if cfg.dataset.name in ['COLLAB', 'REDDIT-BINARY', 'IMDB-BINARY'] and data.x is None:
            # Need to augment X
            data.x = torch.ones((num_nodes, 1))

        if cfg.transform.name is None:
            return data

        cayley_n = self._get_cayley_n(num_nodes)

        # EGP
        if cfg.transform.name == 'EGP':
            data.rewiring_edge_index = self._get_egp_edge_index(cayley_n, num_nodes)
            return data
        elif cfg.transform.name == 'CGP':
            # CGP
            data.rewiring_edge_index, cayley_num_nodes = self._get_cgp_edge_index(cayley_n)

            # Get the number of virtual nodes needed
            virtual_num_nodes = cayley_num_nodes - num_nodes

            # Create a boolean mask to indicate if the node is a virtual node
            data.virtual_node_mask = torch.cat((torch.zeros(num_nodes, dtype=torch.bool), torch.ones(virtual_num_nodes, dtype=torch.bool)), axis=0)

            # Update the input features to have the zero-node embeddings for the virtual nodes
            data.num_nodes = cayley_num_nodes
            data.cayley_num_nodes = cayley_num_nodes
            data.x = torch.cat((data.x, torch.zeros((virtual_num_nodes, data.x.shape[1]), dtype=data.x.dtype)), axis=0)

            return data
        else:
            raise ValueError(f'Expander transform does not exist: {cfg.transform.name}')

    def _get_cayley_n(self, num_nodes):
        n = 1
        while self._cayley_graph_size(n) < num_nodes:
            n += 1
        return n
    
    def _cayley_graph_size(self, n):
        n = int(n)
        return round(n*n*n*np.prod([1 - 1.0/(p * p) for p in list(set(primefac(n)))]))    

    def _get_egp_edge_index(self, cayley_n, num_nodes):
        if cayley_n not in self.cayley_memory:
            self.cayley_memory[cayley_n] = get_cayley_graph(cayley_n)
        
        cayley_graph_edge_index = self.cayley_memory[cayley_n].clone()

        if num_nodes not in self.cayley_node_memory:
            truncated_edge_index = cayley_graph_edge_index[:, torch.logical_and(cayley_graph_edge_index[0] < num_nodes, cayley_graph_edge_index[1] < num_nodes)]
            self.cayley_node_memory[num_nodes] = truncated_edge_index
        
        edge_index = self.cayley_node_memory[num_nodes].clone()

        return edge_index
    
    def _get_cgp_edge_index(self, cayley_n):
        cayley_num_nodes = self._cayley_graph_size(cayley_n)
        if cayley_n not in self.cayley_memory:
            edge_index = get_cayley_graph(cayley_n)
            self.cayley_memory[cayley_n] = edge_index

        edge_index = self.cayley_memory[cayley_n].clone()

        return edge_index, cayley_num_nodes
    
    def _shuffle_edge_index_inplace(self, num_nodes, edge_index):
        perm = np.random.permutation(num_nodes)
        for i in range(edge_index.shape[1]):
            edge_index[0][i] = perm[edge_index[0][i]]
            edge_index[1][i] = perm[edge_index[1][i]]

def get_cayley_graph(n):
    """
        Get the edge index of the Cayley graph (Cay(SL(2, Z_n); S_n)).
    """
    generators = np.array([
        [[1, 1], [0, 1]],
        [[1, n-1], [0, 1]],
        [[1, 0], [1, 1]],
        [[1, 0], [n-1, 1]]])
    ind = 1

    queue = deque([np.array([[1, 0], [0, 1]])])
    nodes = {(1, 0, 0, 1): 0}

    senders = []
    receivers = []

    while queue:
        x = queue.pop()
        x_flat = (x[0][0], x[0][1], x[1][0], x[1][1])
        assert x_flat in nodes
        ind_x = nodes[x_flat]
        for i in range(4):
            tx = np.matmul(x, generators[i])
            tx = np.mod(tx, n)
            tx_flat = (tx[0][0], tx[0][1], tx[1][0], tx[1][1])
            if tx_flat not in nodes:
                nodes[tx_flat] = ind
                ind += 1
                queue.append(tx)
            ind_tx = nodes[tx_flat]

            senders.append(ind_x)
            receivers.append(ind_tx)
    return torch.tensor([senders, receivers])
