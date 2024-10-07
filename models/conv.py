import torch
import torch.nn.functional as F

from utils.config import cfg
from torch_geometric.nn.conv import GINConv

class GNN_node(torch.nn.Module):
    def __init__(self):
        super(GNN_node, self).__init__()

        self.num_layer = cfg.gnn.num_layers
        self.dropout = cfg.gnn.dropout

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layer):
            input_dim = cfg.gnn.input_dim if layer == 0 else cfg.gnn.hidden_dim

            gnn_nn = torch.nn.Sequential(
                torch.nn.Linear(input_dim, cfg.gnn.hidden_dim),
                torch.nn.BatchNorm1d(cfg.gnn.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(cfg.gnn.hidden_dim, cfg.gnn.hidden_dim)
            )
            self.convs.append(GINConv(gnn_nn))

            self.batch_norms.append(torch.nn.BatchNorm1d(cfg.gnn.hidden_dim))
    
    def forward(self, batched_data):
            x, edge_index = batched_data.x, batched_data.edge_index

            h_list = [x.float()]

            for layer in range(self.num_layer):
                h = self.convs[layer](h_list[layer], edge_index)
                h = self.batch_norms[layer](h)

                if layer == self.num_layer - 1:
                    h = F.dropout(h, self.dropout)
                else:
                    h = F.dropout(F.relu(h), self.dropout)

                h_list.append(h)

            node_representation = h_list[-1]

            return node_representation

if __name__ == "__main__":
    GNN_node()
