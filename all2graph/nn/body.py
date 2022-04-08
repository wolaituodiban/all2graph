from copy import deepcopy
from typing import List

import dgl
import torch

from .utils import Module


class Block(Module):
    def __init__(self, conv_layer=None, ff=None, seq_layer=None, ff2=None):
        super().__init__()
        self.conv_layer = conv_layer
        self.ff = ff
        self.seq_layer = seq_layer
        self.ff2 = ff2

    def reset_parameters(self):
        if hasattr(self.conv_layer, 'reset_parameters'):
            self.conv_layer.reset_parameters()
        if hasattr(self.ff, 'reset_parameters'):
            self.seq_layer.reset_parameters()
        if hasattr(self.seq_layer, 'reset_parameters'):
            self.seq_layer.reset_parameters()
        if hasattr(self.ff2, 'reset_parameters'):
            self.seq_layer.reset_parameters()

    def forward(self, graph: dgl.DGLGraph, in_feats, node2seq, seq2node, seq_mask=None) -> torch.Tensor:
        """

        Args:
            graph:
            in_feats:
            node2seq:
            seq2node: 3-tuple
            seq_mask: True才会算
        Returns:

        """
        # view是为了dgl的不同conv
        out_feats = in_feats
        if self.conv_layer is not None:
            out_feats = self.conv_layer(graph, in_feats).view(in_feats.shape[0], -1)
        if self.ff is not None:
            out_feats = self.ff(out_feats)
        # 序列
        if self.seq_layer is not None:
            seq_feats = out_feats[node2seq]
            # 将node2seq中小于0的部分都mask成0
            seq_feats = torch.masked_fill(seq_feats, node2seq.unsqueeze(-1) < 0, 0)
            if seq_mask is None:
                seq_feats = self.seq_layer(seq_feats)
            else:
                seq_feats[seq_mask] += self.seq_layer(seq_feats[seq_mask]) - seq_feats[seq_mask]
            out_feats = seq_feats[seq2node]
        if self.ff2 is not None:
            out_feats = self.ff2(out_feats)
        return out_feats


class Body(Module):
    def __init__(self, num_layers, *args, **kwargs):
        super().__init__()
        block = Block(*args, **kwargs)
        self.layers = torch.nn.ModuleList([deepcopy(block) for _ in range(num_layers)])

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, graph: dgl.DGLGraph, in_feats, node2seq, seq2node, seq_mask=None) -> List[torch.Tensor]:
        """

        Args:
            graph:
            in_feats:
            node2seq:
            seq2node: 3-tuple
            seq_mask: True才会算
        Returns:

        """
        output = []
        out_feats = in_feats
        for layer in self.layers:
            out_feats = layer(graph, out_feats, node2seq=node2seq, seq2node=seq2node, seq_mask=seq_mask)
            output.append(out_feats)
        return output
