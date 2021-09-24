import copy

import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import torch

from .functional import edgewise_linear, nodewise_linear
from .utils import num_parameters
from ..globals import FEATURE, KEY, VALUE, ATTENTION, QUERY, SEP, SRC, WEIGHT, BIAS, DST, NODE


class HeteroAttnConv(torch.nn.Module):
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

    NODE_PARAMS_1D = [QUERY, NODE_BIAS]
    NODE_PARAMS_2D = [NODE_WEIGHT]

    EDGE_PARAMS_1D = [SRC_KEY_BIAS, DST_KEY_BIAS, SRC_VALUE_BIAS, DST_VALUE_BIAS]
    EDGE_PARAMS_2D = [SRC_KEY_WEIGHT, DST_KEY_WEIGHT, SRC_VALUE_WEIGHT, DST_VALUE_WEIGHT]

    def __init__(self, normalized_shape, dropout=0.1, activation='relu', residual=True):
        super().__init__()
        self.key_dropout = torch.nn.Dropout(dropout)
        self.value_dropout = torch.nn.Dropout(dropout)
        self.attn_dropout = torch.nn.Dropout(dropout)
        self.feat_dropout = torch.nn.Dropout(dropout)

        if activation == 'gelu':
            activation = torch.nn.GELU()
        elif activation == 'relu':
            activation = torch.nn.ReLU()
        self.key_activation = copy.deepcopy(activation)
        self.value_activation = copy.deepcopy(activation)
        self.feat_activation = copy.deepcopy(activation)

        if normalized_shape is not None:
            self.feat_norm = torch.nn.LayerNorm(normalized_shape)
        else:
            self.feat_norm = None

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
                QUERY            : (, nhead, out_dim // nhead)
                NODE_WEIGHT      : (, out_dim, in_dim)
                NODE_BIAS        : (, out_dim)
            edata:
                SRC_KEY_WEIGHT   : (, nhead, out_dim // nhead, in_dim)
                DST_KEY_WEIGHT   : (, nhead, out_dim // nhead, in_dim)
                SRC_KEY_BIAS     : (, nhead, out_dim // nhead)
                DST_KEY_BIAS     : (, nhead, out_dim // nhead)

                SRC_VALUE_WEIGHT : (, nhead, out_dim // nhead, in_dim)
                DST_VALUE_WEIGHT : (, nhead, out_dim // nhead, in_dim)
                SRC_VALUE_BIAS   : (, nhead, out_dim // nhead)
                DST_VALUE_BIAS   : (, nhead, out_dim // nhead)
        :param feat: num_nodes * in_dim
        :return:
            node_feat  : (num_nodes, out_dim)
            edge_feat  : (num_edges, out_dim)
            attn_weight: (num_edges, nhead)
        """
        with graph.local_scope():
            # 通过feature计算key
            graph.edata[KEY] = edgewise_linear(
                feat=feat, graph=graph, u_weight=graph.edata[self.SRC_KEY_WEIGHT],
                v_weight=graph.edata[self.DST_KEY_WEIGHT], u_bias=graph.edata[self.SRC_KEY_BIAS],
                v_bias=graph.edata[self.DST_KEY_BIAS], dropout=self.key_dropout,
                activation=self.key_activation)  # (, nheads, out_dim // nheads)

            # 通过feature计算value
            graph.edata[VALUE] = edgewise_linear(
                feat=feat, graph=graph, u_weight=graph.edata[self.SRC_VALUE_WEIGHT],
                v_weight=graph.edata[self.DST_VALUE_WEIGHT], u_bias=graph.edata[self.SRC_VALUE_BIAS],
                v_bias=graph.edata[self.DST_VALUE_BIAS], dropout=self.value_dropout,
                activation=self.value_activation)  # (, nheads, out_dim // nheads)

            # attention
            graph.apply_edges(fn.e_dot_v(KEY, QUERY, ATTENTION))
            graph.edata[ATTENTION] = self.attn_dropout(edge_softmax(graph, graph.edata[ATTENTION]))
            graph.edata[FEATURE] = graph.edata[VALUE] * graph.edata[ATTENTION]
            graph.update_all(fn.copy_e(FEATURE, FEATURE), fn.sum(FEATURE, FEATURE))

            # linear
            node_feat = graph.ndata[FEATURE].view(graph.num_nodes(), -1)
            node_feat = nodewise_linear(
                node_feat, weight=graph.ndata[self.NODE_WEIGHT], bias=graph.ndata[self.NODE_BIAS],
                activation=self.feat_activation
            )
            node_feat = self.feat_dropout(node_feat)
            node_feat = node_feat.view(graph.num_nodes(), -1)
            # add & norm
            if self.residual:
                node_feat = feat + node_feat
            if self.feat_norm is not None:
                node_feat = self.feat_norm(node_feat)

            edge_feat = graph.edata[VALUE].view(graph.num_edges(), -1)
            attn_weight = graph.edata[ATTENTION].view(graph.num_edges(), -1)
            return node_feat, edge_feat, attn_weight

    def extra_repr(self) -> str:
        return 'residual={}, num_parameters={}'.format(self.residual, num_parameters(self))
