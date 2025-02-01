import torch

from utils.config import cfg
from torch_geometric.transforms import BaseTransform

class TuTransform(BaseTransform):
    def __init__(self):
        super(TuTransform).__init__()    

    def __call__(self, data):
        if cfg.dataset.name in ['COLLAB', 'REDDIT-BINARY', 'IMDB-BINARY'] and data.x is None:
            data.x = torch.ones((data.num_nodes, 1))
            
        return data
