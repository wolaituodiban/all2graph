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
                 num_weight: bool = True, key_emb: bool = True, dropout: float = 0.1, conv_config: dict = None,
                 share_layer: bool = False, residual: bool = False, output_configs: dict = None):
        super().__init__()
        self.value_embedding = torch.nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=d_model, **emb_config or {})
        self.node_embedding = NodeEmbedding(embedding_dim=d_model, num_weight=num_weight, key_bias=key_emb)
        self.nhead = nhead
        conv_layer = Conv(normalized_shape=d_model, dropout=dropout, **conv_config or {})
        self.body = Body(
            num_layers=num_layers, conv_layer=conv_layer, share_layer=share_layer, residual=residual)
        self.output = FC(**output_configs or {})
        # assert set(self.dynamic_parameter_shapes) \
        #        == set(self.node_dynamic_parameter_names + self.edge_dynamic_parameter_names)

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
    def node_dynamic_parameter_names(self):
        return self.node_embedding.node_dynamic_parameter_names + self.body.node_dynamic_parameter_names \
               + self.output.node_dynamic_parameter_names

    @property
    def edge_dynamic_parameter_names(self):
        return self.body.edge_dynamic_parameter_names

    @property
    def dynamic_parameter_names_2d(self):
        return self.body.dynamic_parameter_names_2d

    @property
    def dynamic_parameter_names_1d(self):
        return self.node_embedding.dynamic_parameter_names_1d + self.body.dynamic_parameter_names_1d \
               + self.output.dynamic_parameter_names_1d

    @property
    def dynamic_parameter_names_0d(self):
        return self.output.dynamic_parameter_names_0d

    @property
    def dynamic_parameter_shapes(self):
        output = {}
        for name in self.dynamic_parameter_names_0d:
            output[name] = (1, )
        for name in self.dynamic_parameter_names_1d:
            output[name] = (self.nhead, self.d_model // self.nhead)
        for name in self.dynamic_parameter_names_2d:
            output[name] = (self.nhead, self.d_model // self.nhead, self.d_model)
        return output

    def use_matmul(self, b: bool):
        self.body.use_matmul(b)

    def reset_parameters(self):
        self.value_embedding.reset_parameters()
        self.node_embedding.reset_parameters()
        self.body.reset_parameters()

    def forward(self, graph: Graph, emb_param, conv_param, output_params, target_mask, targets):
        value_graph = graph.value_graph.to(self.device)
        value_emb = self.value_embedding(graph.value.to(self.device))
        value_emb = self.node_embedding(value_emb, number=graph.number.to(self.device), parameters=emb_param)
        value_feats, value_keys, value_values, value_attn_weights = self.body(
            graph=value_graph, in_feat=value_emb, parameters=conv_param
        )
        outputs = self.output(feats=value_feats, parameters=output_params, mask=target_mask, targets=targets)
        return outputs, {
            'emb': value_emb,
            'feats': value_feats,
            'keys': value_keys,
            'values': value_values,
            'attn_weights': value_attn_weights
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
