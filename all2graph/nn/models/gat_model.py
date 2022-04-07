import torch

from .model import Model
from ..body import Body
from ..bottle_neck import BottleNeck
from ..embedding import NumEmb
from ..framework import Framework
from ..head import Head


class GATModel(Model):
    def __init__(
            self,
            d_model,
            num_key_layers,
            num_value_layers,
            num_heads,
            out_feats=1,
            dropout=0,
            activation='prelu',
            norm_first=True,
            norm='batch1d',
            gat_kwds=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_key_layers = num_key_layers
        self.num_value_layers = num_value_layers
        self.num_heads = num_heads
        self.out_feats = out_feats
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first
        self.norm = norm
        self.gat_kwds = gat_kwds

    def build_module(self, num_tokens) -> torch.nn.Module:
        module = Framework(
            str_emb=torch.nn.Embedding(num_tokens, self.d_model),
            num_emb=NumEmb(self.d_model, activation=self.activation, norm=self.norm),
            bottle_neck=BottleNeck(
                self.d_model, num_inputs=3, dropout=self.dropout, activation=self.activation,
                norm_first=self.norm_first, norm=self.norm),
            key_emb=Body(
                self.d_model, num_heads=self.num_heads, num_layers=self.num_key_layers, dropout=self.dropout,
                activation=self.activation, norm_first=self.norm_first, norm=self.norm, **self.gat_kwds),
            body=Body(
                self.d_model, num_heads=self.num_heads, num_layers=self.num_value_layers, dropout=self.dropout,
                activation=self.activation, norm_first=self.norm_first, norm=self.norm, **self.gat_kwds),
            head=Head(
                self.d_model, dropout=self.dropout, activation=self.activation, norm_first=self.norm_first,
                out_feats=self.out_feats)
        )
        return module
