import copy
from typing import List, Dict, Tuple

import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import torch

from .functional import edgewise_linear, nodewise_linear
from .utils import num_parameters
from ..globals import FEATURE, ATTENTION, SEP
from ..preserves import KEY, QUERY, SRC, WEIGHT, BIAS, DST, NODE, VALUE


def _get_activation(act):
    if act == 'relu':
        return torch.nn.ReLU()
    elif act == 'gelu':
        return torch.nn.GELU()
    else:
        return copy.deepcopy(act)


class Conv(torch.nn.Module):
    QUERY = QUERY

    SRC_KEY_WEIGHT = SEP.join([SRC, KEY, WEIGHT])
    SRC_KEY_BIAS = SEP.join([SRC, KEY, BIAS])

    DST_KEY_BIAS = SEP.join([DST, KEY, BIAS])
    DST_KEY_WEIGHT = SEP.join([DST, KEY, WEIGHT])

    SRC_VALUE_WEIGHT = SEP.join([SRC, VALUE, WEIGHT])
    SRC_VALUE_BIAS = SEP.join([SRC, VALUE, BIAS])

    DST_VALUE_WEIGHT = SEP.join([DST, VALUE, WEIGHT])
    DST_VALUE_BIAS = SEP.join([DST, VALUE, BIAS])

    NODE_WEIGHT = SEP.join([NODE, WEIGHT])
    NODE_BIAS = SEP.join([NODE, BIAS])

    def __init__(self, normalized_shape, dropout=0.1, key_bias=True, key_norm=False, key_activation=None,
                 value_bias=True, value_norm=False, value_activation=None, node_bias=True, node_norm=False,
                 node_activation='relu', residual=True, norm=True, use_matmul=False):
        super().__init__()
        self.key_dropout = torch.nn.Dropout(dropout)
        self.key_bias = key_bias
        self.key_norm = torch.nn.LayerNorm(normalized_shape) if key_norm else None
        self.key_activation = _get_activation(key_activation)

        self.value_dropout = torch.nn.Dropout(dropout)
        self.value_bias = value_bias
        self.value_norm = torch.nn.LayerNorm(normalized_shape) if value_norm else None
        self.value_activation = _get_activation(value_activation)

        self.attn_dropout = torch.nn.Dropout(dropout)

        # self.node_dropout = torch.nn.Dropout(dropout)
        self.node_bias = node_bias
        self.node_norm = torch.nn.LayerNorm(normalized_shape) if node_norm else None
        self.node_activation = _get_activation(node_activation)

        self.residual = residual
        self.norm = torch.nn.LayerNorm(normalized_shape) if norm else None
        self.use_matmul = use_matmul

    @property
    def edge_dynamic_parameter_names_2d(self):
        return [self.SRC_KEY_WEIGHT, self.DST_KEY_WEIGHT, self.SRC_VALUE_WEIGHT, self.DST_VALUE_WEIGHT]

    @property
    def edge_dynamic_parameter_names_1d(self):
        output = []
        if self.key_bias:
            output += [self.SRC_KEY_BIAS, self.DST_KEY_BIAS]
        if self.value_bias:
            output += [self.SRC_VALUE_BIAS, self.DST_VALUE_BIAS]
        return output

    @property
    def node_dynamic_parameter_names_2d(self):
        return [self.NODE_WEIGHT]

    @property
    def node_dynamic_parameter_names_1d(self):
        output = [self.QUERY]
        if self.node_bias:
            output.append(self.NODE_BIAS)
        return output

    @property
    def node_dynamic_parameter_names(self):
        return self.node_dynamic_parameter_names_2d + self.node_dynamic_parameter_names_1d

    @property
    def edge_dynamic_parameter_names(self):
        return self.edge_dynamic_parameter_names_2d + self.edge_dynamic_parameter_names_1d

    @property
    def dynamic_parameter_names_2d(self):
        return self.node_dynamic_parameter_names_2d + self.edge_dynamic_parameter_names_2d

    @property
    def dynamic_parameter_names_1d(self):
        return self.node_dynamic_parameter_names_1d + self.edge_dynamic_parameter_names_1d

    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(
            self, graph: dgl.DGLGraph, in_feat: torch.Tensor, parameters: Dict[str, torch.Tensor]
    ) -> (torch.Tensor, torch.Tensor):
        """
        K = dot(u, w_uk) + b_uk + dot(v, w_vk) + b_vk
        V = dot(u, w_uv) + b_uv + dot(v, w_vv) + b_vv
        attn_out = softmax(dot(Q, K_T)) * V
        out = dot(attn_out, w_o) + b_o

        :param graph:
        :param in_feat: num_nodes * in_dim
        :param parameters:
        edge weight
            src_key_weight   : (, nhead, out_dim // nhead, in_dim)
            dst_key_weight   : (, nhead, out_dim // nhead, in_dim)
            src_key_bias     : (, nhead, out_dim // nhead)
            dst_key_bias     : (, nhead, out_dim // nhead)

            src_value_weight : (, nhead, out_dim // nhead, in_dim)
            dst_value_weight : (, nhead, out_dim // nhead, in_dim)
            src_value_bias   : (, nhead, out_dim // nhead)
            dst_value_bias   : (, nhead, out_dim // nhead)

        node weight
            query            : (, nhead, out_dim // nhead)
            node_weight      : (, out_dim, in_dim)
            node_bias        : (, out_dim)

        :return:
            out_feat    : (num_nodes, out_dim)
            key_feat    : (num_nodes, out_dim)
            value_feat  : (num_edges, out_dim)
            attn_weight : (num_edges, nhead)
        """
        with graph.local_scope():
            # 通过feature计算key
            graph.edata[KEY] = edgewise_linear(
                feat=in_feat, graph=graph, u_weight=parameters[self.SRC_KEY_WEIGHT],
                v_weight=parameters[self.DST_KEY_WEIGHT],
                u_bias=parameters[self.SRC_KEY_BIAS] if self.key_bias else None,
                v_bias=parameters[self.DST_KEY_BIAS] if self.key_bias else None,
                dropout=self.key_dropout, norm=self.key_norm, activation=self.key_activation
            )  # (, nheads, out_dim // nheads)

            # 通过feature计算value
            graph.edata[VALUE] = edgewise_linear(
                feat=in_feat, graph=graph, u_weight=parameters[self.SRC_VALUE_WEIGHT],
                v_weight=parameters[self.DST_VALUE_WEIGHT],
                u_bias=parameters[self.SRC_VALUE_BIAS] if self.key_bias else None,
                v_bias=parameters[self.DST_VALUE_BIAS] if self.key_bias else None,
                dropout=self.key_dropout, norm=self.key_norm, activation=self.key_activation
            )  # (, nheads, out_dim // nheads)

            # attention
            graph.ndata[QUERY] = parameters[self.QUERY]
            graph.apply_edges(fn.e_dot_v(KEY, QUERY, ATTENTION))
            graph.edata[ATTENTION] = self.attn_dropout(edge_softmax(graph, graph.edata[ATTENTION]))
            graph.edata[FEATURE] = graph.edata[VALUE] * graph.edata[ATTENTION]
            graph.update_all(fn.copy_e(FEATURE, FEATURE), fn.sum(FEATURE, FEATURE))

            # linear
            out_feat = graph.ndata[FEATURE].view(graph.num_nodes(), -1)
            out_feat = nodewise_linear(
                out_feat, weight=parameters[self.NODE_WEIGHT],
                bias=parameters[self.NODE_BIAS] if self.node_bias else None,
                norm=self.node_norm, activation=self.node_activation, use_matmul=self.use_matmul)
            # out_feat = self.node_dropout(out_feat)
            out_feat = out_feat.view(graph.num_nodes(), -1)

            # add & norm
            if self.residual:
                out_feat = in_feat + out_feat

            if self.norm is not None:
                out_feat = self.norm(out_feat)

            key_feat = graph.edata[KEY].view(graph.num_edges(), -1)
            value_feat = graph.edata[VALUE].view(graph.num_edges(), -1)
            attn_weight = graph.edata[ATTENTION].view(graph.num_edges(), -1)
            return out_feat, key_feat, value_feat, attn_weight

    def extra_repr(self) -> str:
        return 'key_bias={}, value_bias={}, node_bias={}, residual={}, num_parameters={}'.format(
            self.key_bias, self.value_bias, self.node_bias, self.residual, num_parameters(self))


class Block(torch.nn.ModuleList):
    def __init__(self, conv_layer: Conv, num_layers: int, share_layer: bool, residual=True):
        if share_layer:
            conv_layers = torch.nn.ModuleList([conv_layer] * num_layers)
        else:
            conv_layers = torch.nn.ModuleList([copy.deepcopy(conv_layer) for _ in range(num_layers)])
        super().__init__(conv_layers)
        self.residual = residual

    @property
    def node_dynamic_parameter_names(self):
        return self[0].node_dynamic_parameter_names

    @property
    def edge_dynamic_parameter_names(self):
        return self[0].edge_dynamic_parameter_names

    @property
    def dynamic_parameter_names_2d(self):
        return self[0].dynamic_parameter_names_2d

    @property
    def dynamic_parameter_names_1d(self):
        return self[0].dynamic_parameter_names_1d

    def reset_parameters(self):
        for layer in self:
            layer.reset_parameters()

    def use_matmul(self, b: bool):
        for conv in self:
            conv.use_matmul = b

    def forward(
            self, graph, in_feat, parameters: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out_feats = []
        keys = []
        values = []
        attn_weights = []
        for conv in self:
            out_feat, key, value, attn_weight = conv(graph, in_feat, parameters)
            out_feats.append(out_feat)
            keys.append(key)
            values.append(value)
            attn_weights.append(attn_weight)
            in_feat = out_feat
        if self.residual:
            out_feats[-1] = out_feats[-1] + in_feat
        out_feats = torch.stack(out_feats, dim=0)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)
        attn_weights = torch.stack(attn_weights, dim=0)
        return out_feats, keys, values, attn_weights

    def extra_repr(self) -> str:
        return 'residual={}, num_parameters={}'.format(self.residual, num_parameters(self))


class Body(torch.nn.ModuleList):
    def __init__(self, conv_layer: Conv, num_layers: List[int], share_layer: bool, residual=True):
        blocks = [Block(conv_layer, n, share_layer=share_layer, residual=residual) for n in num_layers]
        super().__init__(blocks)
        assert set(self.node_dynamic_parameter_names+self.edge_dynamic_parameter_names) ==\
               set(self.dynamic_parameter_names)

    @property
    def node_dynamic_parameter_names(self):
        return self[0].node_dynamic_parameter_names

    @property
    def edge_dynamic_parameter_names(self):
        return self[0].edge_dynamic_parameter_names

    @property
    def dynamic_parameter_names_2d(self):
        return self[0].dynamic_parameter_names_2d

    @property
    def dynamic_parameter_names_1d(self):
        return self[0].dynamic_parameter_names_1d

    @property
    def dynamic_parameter_names(self):
        return self.dynamic_parameter_names_1d + self.dynamic_parameter_names_2d

    def use_matmul(self, b: bool):
        for block in self:
            block.use_matmul(b)

    def reset_parameters(self):
        for layer in self:
            layer.reset_parameters()

    def forward(
            self, graph, in_feat, parameters: Dict[str, torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """

        :param graph:
        :param in_feat:
        :param parameters: dict of tensor with shape(num_blocks, *), 其中*是特定参数的维度
        :return:
            out_feat    : (num_layers, num_nodes, out_dim)
            key_feat    : (num_layers, num_nodes, out_dim)
            value_feat  : (num_layers, num_edges, out_dim)
            attn_weight : (num_layers, num_edges, nhead)
        """
        out_feats = []
        keys = []
        values = []
        attn_weights = []
        for i, conv in enumerate(self):
            out_feat, key, value, attn_weight = conv(graph, in_feat, {k: v[i] for k, v in parameters.items()})
            out_feats.append(out_feat)
            keys.append(key)
            values.append(value)
            attn_weights.append(attn_weight)
            in_feat = out_feat[-1]
        return out_feats, keys, values, attn_weights

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(num_parameters(self))
