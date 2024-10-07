import torch

from utils.config import cfg
from models.conv import GNN_node
from torch_geometric.nn import global_mean_pool

class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()

        self.gnn_node = GNN_node()
        self.pool = global_mean_pool 
        self.graph_pred_linear = torch.nn.Linear(cfg.gnn.hidden_dim, cfg.gnn.output_dim)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        batch_indicator = batched_data.batch

        h_graph = self.pool(h_node, batch_indicator)

        return self.graph_pred_linear(h_graph)

if __name__ == "__main__":
    GNN()
