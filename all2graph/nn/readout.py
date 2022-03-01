from typing import Dict

import torch

from .utils import Module
from ..graph import Graph
from ..globals import VALUE


class Readout(Module):
    def __init__(self, num_feats):
        super().__init__()
        self.linear = torch.nn.Linear(num_feats, 1)

    @property
    def device(self):
        return self.linear.weight.device

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, graph: Graph, feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        output = {}
        for ntype in graph.readout_types:
            readout_feats = graph.push_feats(feats, VALUE, ntype)
            output[ntype] = self.linear(readout_feats).flatten()
        return output
