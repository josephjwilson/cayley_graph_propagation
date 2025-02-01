import logging
from yacs.config import CfgNode as CN

cfg = CN()

def set_cfg(cfg):
    cfg.cfg_dest = 'config.yaml'

    cfg.seed = None
    cfg.device = 0
    cfg.metric = 'ACC'

    cfg.dataset = CN()
    cfg.dataset.format = 'PyG' # OGB, PyG
    cfg.dataset.name = 'MUTAG' # Dataset specific
    cfg.dataset.dir = './datasets'

    cfg.train = CN()
    cfg.train.batch_size = 64
    cfg.train.stopping_patience = 50
    cfg.train.loss_fn = 'cross_entropy' # 'cross_entropy' 'BCE'

    cfg.gnn = CN()
    cfg.gnn.num_layers = 5
    cfg.gnn.hidden_dim = 300
    cfg.gnn.input_dim = None
    cfg.gnn.output_dim = None
    cfg.gnn.layer_type = 'GIN' # GIN, GCN
    cfg.gnn.dropout = 0.5
    cfg.gnn.pool = 'mean'
    cfg.gnn.node_encoder = None # 'Atom', 'Uniform'

    cfg.optim = CN()
    cfg.optim.optimiser = 'adam'
    cfg.optim.base_lr = 0.001
    cfg.optim.max_epochs = 100
    cfg.optim.scheduler = None # 'reduce_on_plateau'
    # Default scheduler parameters
    cfg.optim.scheduler_factor = 0.1
    cfg.optim.scheduler_patience = 10
    cfg.optim.scheduler_min_lr = 0.0

    cfg.transform = CN()
    cfg.transform.name = None

def load_cfg(cfg, args):
    r"""
    Load configurations from file system and command line

    Args:
        cfg (CfgNode): Configuration node
        args (ArgumentParser): Command argument parser

    """
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

def cfg_to_dict(cfg_node, key_list=[]):
    """Convert a config node to dictionary.

    Yacs doesn't have a default function to convert the cfg object to plain
    python dict. The following function was taken from
    https://github.com/rbgirshick/yacs/issues/19
    """
    _VALID_TYPES = {tuple, list, str, int, float, bool}

    if not isinstance(cfg_node, CN):
        if type(cfg_node) not in _VALID_TYPES:
            logging.warning(f"Key {'.'.join(key_list)} with "
                            f"value {type(cfg_node)} is not "
                            f"a valid type; valid types: {_VALID_TYPES}")
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = cfg_to_dict(v, key_list + [k])
        return cfg_dict
