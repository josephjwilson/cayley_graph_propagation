import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import remove_self_loops

class FullyAdjacentTransform(BaseTransform):
    def __init__(self):
        super(FullyAdjacentTransform).__init__()

    def __call__(self, data):
        all_nodes = torch.arange(0, data.num_nodes)
        fully_edge_index =  torch.cartesian_prod(all_nodes, all_nodes).T
        
        # This will work fine in the batching as long as the attribute name contains edge_index
        fully_edge_index_no_sl, _ = remove_self_loops(fully_edge_index)

        data.rewiring_edge_index = fully_edge_index_no_sl

        return data
