import torch
from typing import Callable, Dict, Type

from utils.config import cfg
from models.conv import GNN_node
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

class GNN(torch.nn.Module):
    """
    Graph Neural Network for graph-level prediction tasks.
    
    This model builds on node embeddings to make predictions about entire graphs.
    It uses a configurable pooling mechanism to aggregate node features.
    """
    
    # Map of available pooling functions
    POOLING_MAP: Dict[str, Callable] = {
        'mean': global_mean_pool,
        'add': global_add_pool,
        'max': global_max_pool
    }

    def __init__(self):
        """Initialize the GNN model with node embedding and prediction layers."""
        super(GNN, self).__init__()

        # Node embedding module
        self.gnn_node = GNN_node()
        
        # Configure pooling function based on settings
        self.pool = self._get_pooling_function()
        
        # Graph prediction layer
        self.graph_pred_linear = torch.nn.Linear(cfg.gnn.hidden_dim, cfg.gnn.output_dim)

    def forward(self, batched_data):
        """
        Forward pass through the GNN model.
        
        Args:
            batched_data: Batched graph data object
            
        Returns:
            Graph-level predictions
        """
        # Get node embeddings and batch indicators
        h_node, batch_indicator = self.gnn_node(batched_data)

        # Pool node embeddings to graph embeddings
        h_graph = self.pool(h_node, batch_indicator)

        # Make graph-level predictions
        return self.graph_pred_linear(h_graph)
    
    def _get_pooling_function(self) -> Callable:
        """
        Get the appropriate pooling function based on configuration.
        
        Returns:
            Pooling function to aggregate node features
            
        Raises:
            ValueError: If specified pooling type does not exist
        """
        pooling_type = cfg.gnn.pool.lower()
        
        if pooling_type not in self.POOLING_MAP:
            available_pooling = list(self.POOLING_MAP.keys())
            raise ValueError(f"Invalid pooling type: '{pooling_type}'. "
                             f"Available options: {available_pooling}")
        
        return self.POOLING_MAP[pooling_type]

if __name__ == "__main__":
    GNN()
