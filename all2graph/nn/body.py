from copy import deepcopy
from typing import List

import dgl
import torch

from .utils import Module


class Block(Module):
    def __init__(self, *args, conv_layer=None, ff=None, **kwargs):
        super().__init__()
        self.conv_layer = conv_layer
        self.ff = ff
        self.transformer_layer = torch.nn.TransformerEncoderLayer(*args, **kwargs)

    @property
    def batch_first(self):
        return self.transformer_layer.self_attn.batch_first

    @property
    def squeeze_dim(self):
        if self.batch_first:
            return 0
        else:
            return 1

    def reset_parameters(self):
        if hasattr(self.conv_layer, 'reset_parameters'):
            self.conv_layer.reset_parameters()
        if hasattr(self.ff, 'reset_parameters'):
            self.seq_layer.reset_parameters()
        for module in self.transformer_layer.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, graph: dgl.DGLGraph, in_feats: torch.Tensor, node_masks, src_masks) -> torch.Tensor:
        """

        Args:
            graph:
            in_feats:
            node_masks: mask for each sequence
            src_masks: src_mask for each sequence

        Returns:

        """
        # view是为了dgl的不同conv
        out_feats = in_feats
        if self.conv_layer is not None:
            out_feats = self.conv_layer(graph, in_feats).view(in_feats.shape[0], -1)
        if self.ff is not None:
            out_feats = self.ff(out_feats)
        # 序列
        for node_mask, src_mask in zip(node_masks, src_masks):
            node_mask = node_mask.flatten()
            temp = out_feats[node_mask].unsqueeze(self.squeeze_dim)
            temp = self.transformer_layer(temp, src_mask=src_mask).squeeze(self.squeeze_dim)
            out_feats[node_mask] += temp - out_feats[node_mask]
        return out_feats


class Body(Module):
    def __init__(self, num_layers, *args, conv_layer=None, ff=None, **kwargs):
        super().__init__()
        block = Block(*args, conv_layer=conv_layer, ff=ff, **kwargs)
        self.layers = torch.nn.ModuleList([deepcopy(block) for _ in range(num_layers)])

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, graph: dgl.DGLGraph, in_feats: torch.Tensor, seq_masks: List[torch.Tensor],
                nodes_per_sample: List[int]) -> List[torch.Tensor]:
        output = []
        out_feats = in_feats
        for layer in self.layers:
            out_feats = layer(graph, out_feats, seq_masks, nodes_per_sample)
            output.append(out_feats)
        return output
