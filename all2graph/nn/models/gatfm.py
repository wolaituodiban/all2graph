from typing import Tuple

import dgl.nn.pytorch
import torch

from .model import Model
from ..body import Body
from ..bottle_neck import BottleNeck
from ..embedding import NumEmb
from ..framework import Framework
from ..feedforward import FeedForward
from ..head import Head


class GATFM(Model):
    def __init__(
            self,
            d_model,
            num_layers,
            num_heads,
            seq_degree: Tuple[int, int] = (1, 0),
            out_feats=1,
            dropout=0,
            activation='prelu',
            norm_first=True,
            norm='batch1d',
            gat_kwds=None,
            to_bidirected=False,
            num_featmaps=None,
            head_middle_feats=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, 'invalid num_heads'
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.seq_degree = seq_degree
        self.out_feats = out_feats
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first
        self.norm = norm
        self.gat_kwds = gat_kwds or {'residual': True}
        self.to_bidirected = to_bidirected
        self.num_featmaps = num_featmaps or self.num_layers
        self.head_middle_feats = head_middle_feats or self.d_model

    @property
    def head_middle_feats(self):
        if hasattr(self, '_head_middle_feats'):
            return self._head_middle_feats
        self._head_middle_feats = (self.num_featmaps + 1) * self.d_model
        return self._head_middle_feats

    @head_middle_feats.setter
    def head_middle_feats(self, x):
        self._head_middle_feats = x

    def build_module(self):
        bottle_neck = BottleNeck(
            self.d_model, dropout=self.dropout, activation=self.activation,
            norm_first=self.norm_first, norm=self.norm)
        gat_layer = dgl.nn.pytorch.GATConv(
            in_feats=self.d_model, out_feats=self.d_model // self.num_heads, num_heads=self.num_heads,
            feat_drop=self.dropout, attn_drop=self.dropout, **self.gat_kwds)
        ff = FeedForward(
            self.d_model, activation=self.activation, norm=self.norm, norm_first=self.norm_first,
            dropout=self.dropout, pre=torch.nn.BatchNorm1d(self.d_model))
        body = Body(self.num_layers, conv_layer=gat_layer, ff=ff)
        head = Head(
            (self.num_featmaps + 1) * self.d_model,
            out_feats=self.out_feats,
            middle_feats=self.head_middle_feats,
            activation=self.activation,
        )
        self.module = Framework(
            key_emb=torch.nn.LSTM(self.d_model, self.d_model // 2, 2, bidirectional=True, batch_first=True),
            str_emb=torch.nn.Embedding(self.graph_parser.num_tokens, self.d_model),
            num_emb=NumEmb(self.d_model, activation=self.activation, norm=self.norm),
            bottle_neck=bottle_neck, body=body, head=head,
            seq_degree=self.seq_degree,
            to_bidirected=self.to_bidirected,
            num_featmaps=self.num_featmaps
        )
