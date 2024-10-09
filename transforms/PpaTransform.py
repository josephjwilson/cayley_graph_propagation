import torch

from utils.config import cfg
from torch_geometric.transforms import BaseTransform

class PpaTransform(BaseTransform):
    def __init__(self):
        super(PpaTransform).__init__()    

    def __call__(self, data):
        if cfg.dataset.name == "ogbg-ppa":
            data.x = torch.zeros(data.num_nodes, dtype=torch.long)
            
        return data
