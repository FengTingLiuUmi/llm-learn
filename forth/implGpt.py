import
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self , cfg):
        super().__init__()
        self.att =