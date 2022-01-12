from typing import List

import numpy as np
import torch

from .conv import Conv, Body
from .embedding import NodeEmbedding
from .output import FC
from .utils import num_parameters
from ..graph import Graph


class Encoder(torch.nn.Module):
    """graph factorization machine"""
    def __init__(self, num_embeddings: int, d_model: int, nhead: int, num_layers: List[int], emb_config: dict = None,
                 num_weight: bool = True, key_emb: bool = True, num_activation='prelu', dropout: float = 0.1,
                 conv_config: dict = None, share_layer: bool = False, residual: bool = False,
                 output_config: dict = None):
        super().__init__()
        self.value_embedding = torch.nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=d_model, **emb_config or {})
        self.node_embedding = NodeEmbedding(
            embedding_dim=d_model, num_weight=num_weight, key_bias=key_emb, activation=num_activation)
        self.nhead = nhead
        conv_layer = Conv(normalized_shape=d_model, dropout=dropout, **conv_config or {})
        self.body = Body(
            num_layers=num_layers, conv_layer=conv_layer, share_layer=share_layer, residual=residual)
        self.output = FC(dropout=dropout, **output_config or {})

    @property
    def num_layers(self):
        return list(map(len, self.body))

    @property
    def num_blocks(self):
        return len(self.body)

    @property
    def d_model(self):
        return self.value_embedding.weight.shape[1]

    @property
    def device(self):
        return self.value_embedding.weight.device

    @property
    def node_parameter_names(self):
        return self.node_embedding.node_parameter_names + self.body.node_parameter_names\
               + self.output.node_parameter_names

    @property
    def edge_parameter_names(self):
        return self.body.edge_parameter_names

    @property
    def parameter_names_2d(self):
        return self.body.parameter_names_2d + self.output.parameter_names_2d

    @property
    def parameter_names_1d(self):
        return self.node_embedding.parameter_names_1d + self.body.parameter_names_1d + self.output.parameter_names_1d

    @property
    def parameter_names_0d(self):
        return self.output.parameter_names_0d

    @property
    def parameter_shapes(self):
        output = {}
        for name in self.parameter_names_0d:
            output[name] = (1, )
        for name in self.parameter_names_1d:
            output[name] = (self.nhead, self.d_model // self.nhead)
        for name in self.parameter_names_2d:
            output[name] = (self.nhead, self.d_model // self.nhead, self.d_model)
        return output

    def use_matmul(self, b: bool):
        self.body.use_matmul(b)

    def reset_parameters(self):
        self.value_embedding.reset_parameters()
        self.node_embedding.reset_parameters()
        self.body.reset_parameters()

    def forward(self, graph: Graph, emb_param, conv_param, output_params, target_mask, targets):
        value_emb = self.value_embedding(graph.value)
        value_emb = self.node_embedding(value_emb, number=graph.number, parameters=emb_param)
        value_feats, value_keys, value_values, value_attn_weights = self.body(
            graph=graph.value_graph, in_feat=value_emb, parameters=conv_param
        )
        outputs, hidden_feats = self.output(
            feats=value_feats, parameters=output_params, mask=target_mask, targets=targets)
        return outputs, {
            'emb': value_emb,
            'feats': value_feats,
            'keys': value_keys,
            'values': value_values,
            'attn_weights': value_attn_weights,
            'hidden_feats': hidden_feats
        }

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(num_parameters(self))

    @torch.no_grad()
    def load_pretrained(self, other, self_parser, other_parser):
        state_dict = other.state_dict()
        # 根据parser的string mapper重新映射embedding weight
        emb_weight = self.value_embedding.weight.clone().detach()  # 复制原来的embedding
        load_num = 0  # 加载的参数量
        for word, i in other_parser.string_mapper.items():
            if word in self_parser.string_mapper:
                j = self_parser.string_mapper[word]
                temp = other.value_embedding.weight[i]
                load_num += np.prod(temp.shape)
                emb_weight[j] = temp
        state_dict['value_embedding.weight'] = emb_weight
        self_num = num_parameters(self)
        print('{}: {}/{} ({:.1f}%) loaded from pretrained'.format(
            self.__class__.__name__, load_num, self_num, 100*load_num/self_num))
        self.load_state_dict(state_dict, strict=False)
