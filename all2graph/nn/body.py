from typing import List

import dgl
import torch

from .utils import Module


class Block(Module):
    def __init__(self, conv_layer, seq_layer, ff1=None, ff2=None, batch_first=False):
        super().__init__()
        self.conv_layer = conv_layer
        self.seq_layer = seq_layer
        self.ff1 = ff1
        self.ff2 = ff2
        self.batch_first = batch_first

    def reset_parameters(self):
        if hasattr(self.conv_layer, 'reset_parameters'):
            self.conv_layer.reset_parameters()
        if hasattr(self.seq_layer, 'reset_parameters'):
            self.seq_layer.reset_parameters()
        if hasattr(self.ff1, 'reset_parameters'):
            self.seq_layer.reset_parameters()
        if hasattr(self.ff2, 'reset_parameters'):
            self.seq_layer.reset_parameters()

    def forward(self, graph: dgl.DGLGraph, in_feats: torch.Tensor, seq_ids: List[torch.Tensor]) -> torch.Tensor:
        # view是为了dgl的不同conv
        med_feats = self.conv_layer(graph, in_feats).view(in_feats.shape[0], -1)
        if self.ff1 is not None:
            med_feats = self.ff1(med_feats)
        # 按照seq ids来进行序列计算，为了加总最后的结果，先生成一个全是0的占位符
        output = med_feats
        for ids in seq_ids:
            if self.batch_first:
                dim = 0
            else:
                dim = 1
            temp = med_feats[ids].unsqueeze(dim)
            temp = self.seq_layer(temp)
            # 兼容pytorch的recurrent layers和transformer layers
            if isinstance(temp, tuple):
                temp = temp[0]
            temp = temp.squeeze(dim)
            # print(temp.shape)
            output[ids] += temp - output[ids]
        if self.ff2 is not None:
            output = self.ff2(output)
        return output


class Body(Module):
    def __init__(self, num_layers, conv_layer, seq_layer, ff1=None, ff2=None, batch_first=False):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [Block(conv_layer, seq_layer, ff1, ff2, batch_first) for _ in range(num_layers)])

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, graph: dgl.DGLGraph, in_feats: torch.Tensor, seq_ids: List[torch.Tensor]) -> List[torch.Tensor]:
        output = []
        out_feats = in_feats
        for layer in self.layers:
            out_feats = layer(graph, out_feats, seq_ids)
            output.append(out_feats)
        return output
