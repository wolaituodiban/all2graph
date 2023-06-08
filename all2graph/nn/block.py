import dgl
import torch
from .utils import Module
from .feedforward import FeedForward


class Block(Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, norm_first):
        super().__init__()
        self.conv = dgl.nn.pytorch.EGATConv(
            in_node_feats=d_model, in_edge_feats=d_model,
            out_node_feats=d_model//nhead,
            out_edge_feats=d_model//nhead,
            num_heads=nhead)
        self.nff = FeedForward(
            dim_feedforward, norm_first=norm_first,
            dropout=dropout, pre=torch.nn.BatchNorm1d(dim_feedforward))
        self.enorm = torch.nn.BatchNorm1d(dim_feedforward)
        
    def forward(self, graph: dgl.DGLGraph, nfeats: torch.Tensor, efeats) -> torch.Tensor:
        out_nfeats, out_efeats = self.conv(graph, nfeats, efeats)
        out_nfeats = out_nfeats.view(nfeats.shape[0], -1)
        out_nfeats = out_nfeats + nfeats
        out_nfeats = self.nff(out_nfeats)
        
        out_efeats = out_efeats.view(efeats.shape[0], -1)
        out_efeats = out_efeats + efeats
        out_efeats = self.enorm(out_efeats)
        return out_nfeats, out_efeats