import torch.nn as nn
from abc import ABC

class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
    
    @classmethod
    def from_config(cls, config):
        pass