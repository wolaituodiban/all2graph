from typing import Dict

import torch

from .utils import Module
from ..graph import Graph
from ..globals import VALUE, KEY


class Readout(Module):
    def __init__(self, num_feats):
        super().__init__()
        self.linear = torch.nn.Linear(2 * num_feats, 1)

    @property
    def device(self):
        return self.linear.weight.device

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, graph: Graph, key_feats: torch.Tensor, value_feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        output = {}
        for ntype in graph.readout_types:
            value_feats2 = graph.push_feats(value_feats, VALUE, ntype)
            key_feats2 = graph.push_feats(key_feats, KEY, ntype)
            readout_feats = torch.cat([value_feats2, key_feats2], dim=-1)
            output[ntype] = self.linear(readout_feats).flatten()
        return output
