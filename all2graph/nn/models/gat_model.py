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
            data_parser,
            check_point,
            d_model,
            num_key_layers,
            num_value_layers,
            num_heads,
            out_feats=1,
            dropout=0,
            activation='prelu',
            norm_first=True,
            meta_info_configs=None,
            graph_parser_configs=None,
            post_parser: PostParser = None,
            **kwargs
    ):
        super().__init__(data_parser=data_parser, check_point=check_point, meta_info_configs=meta_info_configs,
                         graph_parser_configs=graph_parser_configs, post_parser=post_parser)
        self.d_model = d_model
        self.num_key_layers = num_key_layers
        self.num_value_layers = num_value_layers
        self.num_heads = num_heads
        self.out_feats = out_feats
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first
        self.kwargs = kwargs

    def build_module(self, num_tokens) -> torch.nn.Module:
        module = Framework(
            token_emb=torch.nn.Embedding(self.graph_parser.num_tokens, self.d_model),
            number_emb=NumEmb(self.d_model, activation=self.activation),
            bottle_neck=BottleNeck(
                self.d_model, dropout=self.dropout, activation=self.activation, norm_first=self.norm_first),
            key_body=GATBody(
                self.d_model, num_heads=self.num_heads, num_layers=self.num_key_layers, dropout=self.dropout,
                activation=self.activation, norm_first=self.norm_first, **self.kwargs),
            value_body=GATBody(
                self.d_model, num_heads=self.num_heads, num_layers=self.num_value_layers, dropout=self.dropout,
                activation=self.activation, norm_first=self.norm_first, **self.kwargs),
            readout=Readout(
                self.d_model, dropout=self.dropout, activation=self.activation, norm_first=self.norm_first,
                out_feats=self.out_feats)
        )
        return module
