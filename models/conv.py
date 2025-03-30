import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Type, Union

from utils.config import cfg
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn.conv import GINConv, GCNConv, MessagePassing

class GNN_node(torch.nn.Module):
    """
    Graph Neural Network node embedding model.
    
    This model handles the node-level representation learning for graph neural networks,
    supporting different layer types (GIN, GCN) and different graph transformations
    (EGP, CGP, FA).
    """
    
    def __init__(self):
        """Initialize the GNN node model with layers defined in config."""
        super(GNN_node, self).__init__()

        # Model configuration
        self.num_layers = cfg.gnn.num_layers
        self.dropout = cfg.gnn.dropout
        self.hidden_dim = cfg.gnn.hidden_dim

        # Node encoder for initial embeddings
        self.node_encoder = self._get_node_encoder(self.hidden_dim)

        # Initialize layer collections
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # Create layers
        for layer in range(self.num_layers):
            # Determine input dimension for the current layer
            input_dim = self._get_layer_input_dim(layer)
            
            # Add GNN layer based on configuration
            self.convs.append(self._get_layer(input_dim, self.hidden_dim))
            
            # Add batch normalization
            self.batch_norms.append(torch.nn.BatchNorm1d(self.hidden_dim))

    def forward(self, batched_data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the GNN node model.
        
        Args:
            batched_data: Batched graph data object with node features and edge indices
            
        Returns:
            Tuple containing:
            - Node representations after message passing
            - Batch indicator tensor for pooling
        """
        # Get initial node embeddings
        x = self._get_initial_node_embeddings(batched_data)
        batch = batched_data.batch
        
        # Handle Cayley Graph Propagation (CGP) transformation
        is_cgp = cfg.transform.name == "CGP"
        if is_cgp:
            x = self._prepare_cgp_embeddings(x, batched_data)
        
        # Initialize hidden state list with initial embeddings
        h_list = [x]
        
        # Apply GNN layers
        for layer in range(self.num_layers):
            # Determine which edge index to use based on layer and transform type
            edge_index = self._get_edge_index_for_layer(layer, batched_data)
            
            # Apply convolution, batch norm, and activation
            h = self.convs[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)
            
            # Apply dropout (and ReLU for non-final layers)
            if layer == self.num_layers - 1:
                h = F.dropout(h, self.dropout)
            else:
                h = F.dropout(F.relu(h), self.dropout)
            
            h_list.append(h)
        
        # Extract final node representations
        node_representation = h_list[-1]
        
        # Filter out virtual nodes for CGP
        if is_cgp:
            node_representation = node_representation[~batched_data.virtual_node_mask]
            batch = batch[~batched_data.virtual_node_mask]
            
        return node_representation, batch
    
    def _get_initial_node_embeddings(self, batched_data) -> torch.Tensor:
        """Extract and process initial node features from the data."""
        if self.node_encoder is not None:
            return self.node_encoder(batched_data.x)
        return batched_data.x.float()
    
    def _prepare_cgp_embeddings(self, x: torch.Tensor, batched_data) -> torch.Tensor:
        """Prepare embeddings for Cayley Graph Propagation by handling virtual nodes."""
        x_embeddings = torch.zeros((x.shape[0], x.shape[1]), device=x.device)
        x_embeddings[~batched_data.virtual_node_mask] = x[~batched_data.virtual_node_mask]
        return x_embeddings
    
    def _get_edge_index_for_layer(self, layer: int, batched_data) -> torch.Tensor:
        """
        Determine which edge index to use for the given layer based on configuration.
        
        The behavior is controlled by cfg.transform.rewiring_schedule:
        - 'base': Always use the original edge_index (default)
        - 'rewired': Always use the rewired edge_index for all layers
        - 'interweave': Alternate between original and rewired edges (odd layers use rewired)
        - 'last_layer': Use original edges for all layers except the final one
        
        Returns:
            The edge index to use for the current layer
        """
        # If no transformation applied, always use the original edge_index
        if cfg.transform.name is None:
            return batched_data.edge_index
            
        # Get the rewiring schedule (default to 'base' if not specified)
        rewiring_schedule = cfg.transform.rewiring_schedule.lower() if hasattr(cfg.transform, 'rewiring_schedule') else 'base'
        
        if rewiring_schedule == 'rewired':
            # Always use rewired edges for all layers
            return batched_data.rewiring_edge_index
        elif rewiring_schedule == 'interweave':
            # Alternate between original and rewired (odd layers use rewired)
            if layer % 2 == 1:
                return batched_data.rewiring_edge_index
        elif rewiring_schedule == 'last_layer':
            # Use rewired only in the final layer
            if layer == self.num_layers - 1:
                return batched_data.rewiring_edge_index
        
        # Default: use original edges
        return batched_data.edge_index

    def _get_layer_input_dim(self, layer: int) -> int:
        """Determine the input dimension for a given layer."""
        # First layer has special input dimension handling
        if layer == 0:
            if cfg.gnn.node_encoder == 'Atom':
                return self.hidden_dim
            else:
                return cfg.gnn.input_dim
        # All other layers use hidden_dim as input
        return self.hidden_dim

    def _get_node_encoder(self, hidden_dim: int) -> Optional[torch.nn.Module]:
        """
        Get the appropriate node encoder based on configuration.
        
        Args:
            hidden_dim: Dimension of hidden representations
            
        Returns:
            Node encoder module or None if no encoder is configured
            
        Raises:
            ValueError: If specified node encoder does not exist
        """
        if cfg.gnn.node_encoder is None:
            return None
        
        node_encoder = cfg.gnn.node_encoder.lower()

        if node_encoder == 'atom':
            return AtomEncoder(hidden_dim)
        elif node_encoder == 'uniform':
            return torch.nn.Embedding(1, hidden_dim)
        else:
            raise ValueError(f'Node encoder does not exist: {node_encoder}')

    def _get_layer(self, input_dim: int, hidden_dim: int) -> MessagePassing:
        """
        Get the appropriate GNN layer based on configuration.
        
        Args:
            input_dim: Input dimension for the layer
            hidden_dim: Output dimension for the layer
            
        Returns:
            Configured GNN layer
        """
        gnn_type = cfg.gnn.layer_type.lower()
        if gnn_type == 'gin':
            return self._get_gin_layer(input_dim, hidden_dim)
        elif gnn_type == 'gcn':
            return GCNConv(input_dim, hidden_dim)
        else:
            raise ValueError(f'Layer type does not exist: {gnn_type}')

    def _get_gin_layer(self, input_dim: int, hidden_dim: int) -> GINConv:
        """
        Create a Graph Isomorphism Network (GIN) layer.
        
        Args:
            input_dim: Input dimension for the layer
            hidden_dim: Output dimension for the layer
            
        Returns:
            Configured GIN layer
        """
        # For OGB datasets, use double the hidden dim for the intermediate MLP layer
        emb_dim = 2 * hidden_dim if cfg.dataset.format.lower() == 'ogb' else hidden_dim

        # Create MLP with skip connection
        mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, emb_dim),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, hidden_dim)
        )

        return GINConv(mlp)

if __name__ == "__main__":
    GNN_node()
