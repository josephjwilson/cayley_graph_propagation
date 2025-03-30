import logging
from typing import Any, Dict, List, Union
from yacs.config import CfgNode as CN
from argparse import Namespace

# Global configuration object
cfg = CN()

def set_default_cfg(cfg: CN) -> None:
    """
    Set default configuration values.
    
    Args:
        cfg: Configuration node to initialize with default values
    """
    # Core settings
    cfg.cfg_dest = 'config.yaml'
    cfg.seed = None
    cfg.device = 0
    cfg.metric = 'ACC'

    # Dataset configuration
    _configure_dataset(cfg)
    
    # Training configuration
    _configure_training(cfg)
    
    # GNN model configuration
    _configure_gnn(cfg)
    
    # Optimization configuration
    _configure_optimization(cfg)
    
    # Transform configuration
    _configure_transform(cfg)

def _configure_dataset(cfg: CN) -> None:
    """Configure dataset-related settings"""
    cfg.dataset = CN()
    cfg.dataset.format = 'PyG'  # OGB, PyG
    cfg.dataset.name = 'MUTAG'  # Dataset specific
    cfg.dataset.dir = './datasets'

def _configure_training(cfg: CN) -> None:
    """Configure training-related settings"""
    cfg.train = CN()
    cfg.train.batch_size = 64
    cfg.train.stopping_patience = 50
    cfg.train.loss_fn = 'cross_entropy'  # Alternatives: 'BCE'

def _configure_gnn(cfg: CN) -> None:
    """Configure GNN model architecture settings"""
    cfg.gnn = CN()
    cfg.gnn.num_layers = 5
    cfg.gnn.hidden_dim = 300
    cfg.gnn.input_dim = None  # Auto-configured based on dataset
    cfg.gnn.output_dim = None  # Auto-configured based on dataset
    cfg.gnn.layer_type = 'GIN'  # Alternatives: 'GCN'
    cfg.gnn.dropout = 0.5
    cfg.gnn.pool = 'mean'
    cfg.gnn.node_encoder = None  # Alternatives: 'Atom', 'Uniform'

def _configure_optimization(cfg: CN) -> None:
    """Configure optimization-related settings"""
    cfg.optim = CN()
    cfg.optim.optimiser = 'adam'
    cfg.optim.base_lr = 0.001
    cfg.optim.max_epochs = 100
    cfg.optim.scheduler = None  # Alternatives: 'reduce_on_plateau'
    
    # Scheduler parameters (used when scheduler is enabled)
    cfg.optim.scheduler_factor = 0.1
    cfg.optim.scheduler_patience = 10
    cfg.optim.scheduler_min_lr = 0.0

def _configure_transform(cfg: CN) -> None:
    """Configure graph transformation settings"""
    cfg.transform = CN()
    cfg.transform.name = None  # Alternatives: 'EGP', 'CGP', 'FA', 'DIGL', 'SDRF', 'BORF', 'GTR', 'FoSR'
    
    # DIGL parameters
    cfg.transform.alpha = 0.1     # Teleport probability in PPR computation
    cfg.transform.k = 128         # Number of neighbors to keep per node
    cfg.transform.eps = None      # Threshold for edge clipping
    
    # SDRF/BORF common parameters
    cfg.transform.loops = 10            # Number of rewiring iterations
    cfg.transform.remove_edges = True   # Whether to also remove high-curvature edges
    cfg.transform.removal_bound = 0.5   # Minimum curvature for edge removal
    cfg.transform.tau = 1               # Temperature parameter for softmax sampling
    cfg.transform.is_undirected = False # Whether the graph is undirected
    
    # BORF-specific parameters
    cfg.transform.batch_add = 4          # Number of edges to add in each iteration
    cfg.transform.batch_remove = 2       # Number of edges to remove in each iteration
    cfg.transform.algorithm = 'borf3'    # BORF algorithm to use ('borf2' or 'borf3')
    
    # GTR parameters
    cfg.transform.num_edges = 10      # Number of edges to add to the graph
    cfg.transform.try_gpu = True      # Whether to use GPU acceleration if available
    
    # FoSR parameters
    cfg.transform.num_iterations = 100    # Number of iterations for spectral rewiring
    cfg.transform.initial_power_iters = 10  # Number of power iterations for initialization

def load_cfg(cfg: CN, args: Namespace) -> None:
    """
    Load configurations from file system and command line.
    
    Args:
        cfg: Configuration node to update
        args: Command line arguments
    """
    # Load config from file first, then override with command line options
    cfg.merge_from_file(args.cfg_file)
    
    if args.opts:
        cfg.merge_from_list(args.opts)

def cfg_to_dict(cfg_node: Union[CN, Any], key_list: List[str] = []) -> Dict[str, Any]:
    """
    Convert a config node to a dictionary.
    
    Based on: https://github.com/rbgirshick/yacs/issues/19
    
    Args:
        cfg_node: Config node or value to convert
        key_list: List of keys in current position (for error reporting)
        
    Returns:
        Dict representation of the config
    """
    _VALID_TYPES = {tuple, list, str, int, float, bool}

    # Handle leaf nodes (non-CN types)
    if not isinstance(cfg_node, CN):
        if type(cfg_node) not in _VALID_TYPES:
            logging.warning(
                f"Key {'.'.join(key_list)} with value type {type(cfg_node)} "
                f"is not a valid type; valid types: {_VALID_TYPES}"
            )
        return cfg_node
    
    # Handle CN nodes by recursively converting their contents
    cfg_dict = dict(cfg_node)
    for k, v in cfg_dict.items():
        cfg_dict[k] = cfg_to_dict(v, key_list + [k])
    
    return cfg_dict
    
# Alias for backward compatibility
set_cfg = set_default_cfg
