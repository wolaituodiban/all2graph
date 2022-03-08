import torch

from .model import Model
from ..body import GATBody
from ..bottle_neck import BottleNeck
from ..embedding import NumEmb
from ..framework import Framework
from ..readout import Readout
from ...parsers import PostParser


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
        self.gat_kwds = gat_kwds

    def build_module(self, num_tokens) -> torch.nn.Module:
        module = Framework(
            token_emb=torch.nn.Embedding(num_tokens, self.d_model),
            number_emb=NumEmb(self.d_model, activation=self.activation),
            bottle_neck=BottleNeck(
                self.d_model, num_inputs=3, dropout=self.dropout, activation=self.activation, norm_first=self.norm_first),
            key_body=GATBody(
                self.d_model, num_heads=self.num_heads, num_layers=self.num_key_layers, dropout=self.dropout,
                activation=self.activation, norm_first=self.norm_first, **self.gat_kwds),
            value_body=GATBody(
                self.d_model, num_heads=self.num_heads, num_layers=self.num_value_layers, dropout=self.dropout,
                activation=self.activation, norm_first=self.norm_first, **self.gat_kwds),
            readout=Readout(
                self.d_model, dropout=self.dropout, activation=self.activation, norm_first=self.norm_first,
                out_feats=self.out_feats)
        )
        return module
