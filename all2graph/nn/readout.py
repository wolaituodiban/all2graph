from typing import Dict

import torch

from .bottle_neck import BottleNeck
from .utils import Module


class Readout(Module):
    def __init__(self, in_feats, out_feats=1, dropout=0, activation='prelu', norm_first=True):
        super().__init__()
        self.bottle_neck = BottleNeck(
            d_model=in_feats, dropout=dropout, activation=activation, norm='batch1d', norm_first=norm_first)
        self.output = torch.nn.Linear(in_feats, out_feats)

    @property
    def device(self):
        return self.linear.weight.device

    def reset_parameters(self):
        self.bottle_neck.reset_parameters()
        self.output.reset_parameters()

    def forward(self, key_feats, value_feats) -> Dict[str, torch.Tensor]:
        readout_feats = self.bottle_neck(key_feats, value_feats)
        return self.output(readout_feats).squeeze(-1)
