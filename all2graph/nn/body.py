import dgl
from dgl.nn.pytorch import GATConv
import torch
from .feedforward import FeedForward
from .utils import Module, _get_activation


class Body(Module):
    def forward(self, graph: dgl.DGLHeteroGraph, in_feats: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GATBody(Body):
    def __init__(self, num_feats, num_heads, num_layers, dropout=0, activation='relu', norm_first=True, **kwargs):
        super().__init__()
        assert num_feats % num_heads == 0
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GATConv(
                    num_feats, num_feats // num_heads, num_heads, activation=_get_activation(activation),
                    feat_drop=dropout, attn_drop=dropout, **kwargs)
            ff = FeedForward(num_feats, dropout=dropout, activation=activation, norm_first=norm_first)
            self.layers.append(torch.nn.ModuleDict({'conv': conv, 'ff': ff}))

    @property
    def device(self):
        return self.ff[0].device

    def reset_parameters(self):
        for layer in self.layers:
            layer['conv'].reset_parameters()
            layer['ff'].reset_parameters()

    def set_allow_zero_in_degree(self, *args, **kwargs):
        for conv in self.conv:
            conv.set_allow_zero_in_degree(*args, **kwargs)

    def forward(self, graph: dgl.DGLHeteroGraph, in_feats: torch.Tensor) -> torch.Tensor:
        out_feats = in_feats
        for layer in self.layers:
            out_feats = layer['conv'](graph, out_feats)
            out_feats = layer['ff'](out_feats.view(out_feats.shape[0], -1))
        return out_feats
