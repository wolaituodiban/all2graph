from typing import Dict

import torch

from .bottle_neck import BottleNeck
from .utils import Module
from ..graph import Graph


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

    def forward(self, graph: Graph, key_feats: torch.Tensor, value_feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        output = {}
        for ntype in graph.readout_types:
            readout_feats = self.bottle_neck(
                graph.push_key2readout(key_feats, ntype), graph.push_value2readout(value_feats, ntype))
            output[ntype] = self.output(readout_feats).squeeze(-1)
        return output
