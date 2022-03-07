from typing import List
import dgl
from dgl.nn.pytorch import GATConv
import torch
from .feedforward import FeedForward
from .utils import Module, _get_activation, _get_norm


class Body(Module):
    def forward(self, graph: dgl.DGLHeteroGraph, in_feats: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError


class GATBody(Body):
    def __init__(self, num_feats, num_heads, num_layers, dropout=0, activation='prelu', norm='layer', norm_first=True,
                 **kwargs):
        super().__init__()
        assert num_feats % num_heads == 0
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GATConv(
                    num_feats, num_feats // num_heads, num_heads, activation=_get_activation(activation),
                    feat_drop=dropout, attn_drop=dropout, **kwargs)
            norm = _get_norm(norm, num_feats)
            ff = FeedForward(num_feats, dropout=dropout, activation=activation, norm=norm, norm_first=norm_first)
            self.layers.append(torch.nn.ModuleDict({'conv': conv, 'norm': norm, 'ff': ff}))

    @property
    def device(self):
        return self.ff[0].device

    def reset_parameters(self):
        for layer in self.layers:
            layer['conv'].reset_parameters()
            layer['norm'].reset_parameters()
            layer['ff'].reset_parameters()

    def set_allow_zero_in_degree(self, *args, **kwargs):
        for conv in self.conv:
            conv.set_allow_zero_in_degree(*args, **kwargs)

    def forward(self, graph: dgl.DGLHeteroGraph, in_feats: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        temp_feats = in_feats
        for layer in self.layers:
            temp_feats = layer['conv'](graph, temp_feats)
            temp_feats = temp_feats.view(temp_feats.shape[0], -1)
            temp_feats = layer['norm'](temp_feats)
            temp_feats = layer['ff'](temp_feats)
            outputs.append(temp_feats)
        return outputs
