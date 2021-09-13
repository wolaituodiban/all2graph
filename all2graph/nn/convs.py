import copy

import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import torch

from .functional import edgewise_linear, nodewise_linear
from ..globals import FEATURE, KEY, VALUE, ATTENTION, \
    SRC_KEY_BIAS, SRC_KEY_WEIGHT, DST_KEY_BIAS, DST_KEY_WEIGHT, \
    SRC_VALUE_BIAS, SRC_VALUE_WEIGHT, DST_VALUE_BIAS, DST_VALUE_WEIGHT, \
    NODE_WEIGHT, NODE_BIAS, QUERY


class Conv(torch.nn.Module):
    def __init__(self, normalized_shape, dropout=0.1, activation='relu', residual=True):
        super().__init__()
        if normalized_shape is not None:
            self.out_norm = torch.nn.LayerNorm(normalized_shape)
        else:
            self.out_norm = None

        self.key_dropout = torch.nn.Dropout(dropout)
        self.value_dropout = torch.nn.Dropout(dropout)
        self.attn_dropout = torch.nn.Dropout(dropout)
        self.out_dropout = torch.nn.Dropout(dropout)

        if activation == 'gelu':
            activation = torch.nn.GELU()
        elif activation == 'relu':
            activation = torch.nn.ReLU()
        self.key_activation = copy.deepcopy(activation)
        self.value_activation = copy.deepcopy(activation)
        self.out_activation = copy.deepcopy(activation)

        self.residual = residual

    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        K = dot(u, w_uk) + b_uk + dot(v, w_vk) + b_vk
        V = dot(u, w_uv) + b_uv + dot(v, w_vv) + b_vv
        attn_out = softmax(dot(Q, K_T)) * V
        out = dot(attn_out, w_o) + b_o
        :param graph:
            ndata:
                QUERY            : (, nheads, out_dim // nheads)
                NODE_WEIGHT      : (, out_dim, in_dim)
                NODE_BIAS        : (, out_dim)
            edata:
                SRC_KEY_WEIGHT   : (, nheads, out_dim // nheads, in_dim)
                DST_KEY_WEIGHT   : (, nheads, out_dim // nheads, in_dim)
                SRC_KEY_BIAS     : (, nheads, out_dim // nheads)
                DST_KEY_BIAS     : (, nheads, out_dim // nheads)

                SRC_VALUE_WEIGHT : (, nheads, out_dim // nheads, in_dim)
                DST_VALUE_WEIGHT : (, nheads, out_dim // nheads, in_dim)
                SRC_VALUE_BIAS   : (, nheads, out_dim // nheads)
                DST_VALUE_BIAS   : (, nheads, out_dim // nheads)
        :param feat: num_nodes * in_dim
        :return:
            FEATURE  : (num_nodes, out_dim)
            KEY      : (num_edges, out_dim)
            VALUE    : (num_edges, out_dim)
            ATTENTION: (num_edges, nheads)
        """
        with graph.local_scope():
            # 通过feature计算key
            graph.edata[KEY] = edgewise_linear(
                feat=feat, graph=graph, u_weight=graph.edata[SRC_KEY_WEIGHT], v_weight=graph.edata[DST_KEY_WEIGHT],
                u_bias=graph.edata[SRC_KEY_BIAS], v_bias=graph.edata[DST_KEY_BIAS], dropout=self.key_dropout,
                activation=self.key_activation)  # (, nheads, out_dim // nheads)

            # 通过feature计算value
            graph.edata[VALUE] = edgewise_linear(
                feat=feat, graph=graph, u_weight=graph.edata[SRC_VALUE_WEIGHT], v_weight=graph.edata[DST_VALUE_WEIGHT],
                u_bias=graph.edata[SRC_VALUE_BIAS], v_bias=graph.edata[DST_VALUE_BIAS], dropout=self.value_dropout,
                activation=self.value_activation)  # (, nheads, out_dim // nheads)

            # attention
            graph.apply_edges(fn.e_dot_v(KEY, QUERY, ATTENTION))
            graph.edata[ATTENTION] = self.attn_dropout(edge_softmax(graph, graph.edata[ATTENTION]))
            graph.edata[FEATURE] = graph.edata[VALUE] * graph.edata[ATTENTION]
            graph.update_all(fn.copy_e(FEATURE, FEATURE), fn.sum(FEATURE, FEATURE))

            # linear
            out_feat = graph.ndata[FEATURE].view(graph.num_nodes(), -1)
            out_feat = nodewise_linear(
                out_feat, weight=graph.ndata[NODE_WEIGHT], bias=graph.ndata[NODE_BIAS],
                activation=self.out_activation
            )
            out_feat = self.out_dropout(out_feat)
            out_feat = out_feat.view(graph.num_nodes(), -1)
            # add & norm
            if self.residual:
                out_feat += feat
            if self.out_norm is not None:
                out_feat = self.out_norm(out_feat)

            key_feat = graph.edata[KEY].view(graph.num_edges(), -1)
            value_feat = graph.edata[VALUE].view(graph.num_edges(), -1)
            attn_weight = graph.edata[ATTENTION].view(graph.num_edges(), -1)
            return out_feat, key_feat, value_feat, attn_weight
