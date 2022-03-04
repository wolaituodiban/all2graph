from typing import Dict

import torch

from .utils import Module, _get_activation
from ..graph import Graph


class Readout(Module):
    def __init__(self, in_feats, out_feats=1, dropout=0, activation='relu', norm_first=True):
        super().__init__()
        dropout = torch.nn.Dropout(dropout)
        linear = torch.nn.Linear(2 * in_feats, in_feats)
        norm = torch.nn.LayerNorm(in_feats)
        activation = _get_activation(activation)
        output = torch.nn.Linear(in_feats, out_feats)
        if norm_first:
            self.layers = torch.nn.Sequential(dropout, linear, norm, activation, output)
        else:
            self.layers = torch.nn.Sequential(dropout, linear, activation, norm, output)

    @property
    def device(self):
        return self.linear.weight.device

    def reset_parameters(self):
        for module in self.layers:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, graph: Graph, key_feats: torch.Tensor, value_feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        output = {}
        for ntype in graph.readout_types:
            value_feats2 = graph.push_value2readout(value_feats, ntype)
            key_feats2 = graph.push_key2readout(key_feats, ntype)
            readout_feats = torch.cat([value_feats2, key_feats2], dim=-1)
            output[ntype] = self.layers(readout_feats).squeeze(-1)
        return output
