from copy import deepcopy
from typing import List

import dgl
import torch

from .utils import Module


class Block(Module):
    def __init__(self, conv_layer=None, ff=None, seq_layer=None, ff2=None, transpose_dim=None):
        super().__init__()
        self.conv_layer = conv_layer
        self.ff = ff
        self.seq_layer = seq_layer
        self.ff2 = ff2
        self.transpose_dim = transpose_dim

    def reset_parameters(self):
        if hasattr(self.conv_layer, 'reset_parameters'):
            self.conv_layer.reset_parameters()
        if hasattr(self.ff, 'reset_parameters'):
            self.seq_layer.reset_parameters()
        if hasattr(self.seq_layer, 'reset_parameters'):
            self.seq_layer.reset_parameters()
        if hasattr(self.ff2, 'reset_parameters'):
            self.seq_layer.reset_parameters()

    def conv_forward(self, graph, in_feats):
        out_feats = in_feats
        if self.conv_layer is not None:
            out_feats = self.conv_layer(graph, in_feats).view(in_feats.shape[0], -1)
        if self.ff is not None:
            out_feats = self.ff(out_feats)
        return out_feats

    def seq_layer_forward(self, in_feats):
        if self.transpose_dim is not None:
            in_feats = in_feats.transpose(*self.transpose_dim)
        out_feats = self.seq_layer(in_feats)
        if isinstance(out_feats, tuple):
            out_feats = out_feats[0]
        if self.transpose_dim is not None:
            out_feats = out_feats.transpose(*self.transpose_dim)
        return out_feats

    def seq_forward(self, in_feats, node2seq, seq2node, seq_mask):
        out_feats = in_feats
        if self.seq_layer is not None:
            seq_feats = out_feats[node2seq]
            # 将node2seq中小于0的部分都mask成0
            seq_feats = torch.masked_fill(seq_feats, node2seq.unsqueeze(-1) < 0, 0)
            if seq_mask is None:
                seq_feats = self.seq_layer_forward(seq_feats)
            else:
                seq_feats[seq_mask] += self.seq_layer_forward(seq_feats[seq_mask]) - seq_feats[seq_mask]
            out_feats = seq_feats[seq2node]
        if self.ff2 is not None:
            out_feats = self.ff2(out_feats)
        return out_feats

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
        out_feats = self.conv_forward(graph, in_feats)
        return self.seq_forward(out_feats, node2seq=node2seq, seq2node=seq2node, seq_mask=seq_mask)


class Body(Module):
    def __init__(self, num_layers, conv_layer=None, ff=None, seq_layer=None, ff2=None, transpose_dim=None,
                 conv_first=True, conv_last=True):
        super().__init__()
        block = Block(conv_layer=conv_layer, ff=ff, seq_layer=seq_layer, ff2=ff2, transpose_dim=transpose_dim)
        blocks = [deepcopy(block) for _ in range(num_layers)]
        if not conv_first:
            blocks[0].conv_layer = None
            blocks[0].ff = None
        if conv_last:
            blocks[-1].seq_layer = None
            blocks[-1].ff2 = None
        self.layers = torch.nn.ModuleList(blocks)

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
