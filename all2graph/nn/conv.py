import copy

import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import torch

from .functional import edgewise_linear, nodewise_linear
from .utils import num_parameters
from ..globals import FEATURE, KEY, VALUE, ATTENTION, QUERY, SEP, SRC, WEIGHT, BIAS, DST, NODE


def _get_activation(act):
    if act == 'relu':
        return torch.nn.ReLU()
    elif act == 'gelu':
        return torch.nn.GELU()
    else:
        return copy.deepcopy(act)


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

    def __init__(self, normalized_shape, dropout=0.1, key_norm=False, key_activation=None,
                 value_norm=False, value_activation=None, node_norm=False, node_activation='relu',
                 residual=True, norm=True):
        super().__init__()
        self.key_dropout = torch.nn.Dropout(dropout)
        self.value_dropout = torch.nn.Dropout(dropout)

        self.key_norm = torch.nn.LayerNorm(normalized_shape) if key_norm else None
        self.key_activation = _get_activation(key_activation)

        self.value_norm = torch.nn.LayerNorm(normalized_shape) if value_norm else None
        self.value_activation = _get_activation(value_activation)

        self.attn_dropout = torch.nn.Dropout(dropout)

        # self.node_dropout = torch.nn.Dropout(dropout)
        self.node_norm = torch.nn.LayerNorm(normalized_shape) if node_norm else None
        self.node_activation = _get_activation(node_activation)

        self.residual = residual
        self.norm = torch.nn.LayerNorm(normalized_shape) if norm else None

    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, graph: dgl.DGLGraph, in_feat: torch.Tensor) -> (torch.Tensor, torch.Tensor):
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
        :param in_feat: num_nodes * in_dim
        :return:
            node_feat  : (num_nodes, out_dim)
            edge_feat  : (num_edges, out_dim)
            attn_weight: (num_edges, nhead)
        """
        with graph.local_scope():
            # 通过feature计算key
            graph.edata[KEY] = edgewise_linear(
                feat=in_feat, graph=graph, u_weight=graph.edata[self.SRC_KEY_WEIGHT],
                v_weight=graph.edata[self.DST_KEY_WEIGHT], u_bias=getattr(graph.edata, self.SRC_KEY_BIAS, None),
                v_bias=getattr(graph.edata, self.DST_KEY_BIAS, None), dropout=self.key_dropout,
                norm=self.key_norm, activation=self.key_activation)  # (, nheads, out_dim // nheads)

            # 通过feature计算value
            graph.edata[VALUE] = edgewise_linear(
                feat=in_feat, graph=graph, u_weight=graph.edata[self.SRC_VALUE_WEIGHT],
                v_weight=graph.edata[self.DST_VALUE_WEIGHT], u_bias=getattr(graph.edata, self.SRC_VALUE_BIAS, None),
                v_bias=getattr(graph.edata, self.DST_VALUE_BIAS, None), dropout=self.value_dropout,
                norm=self.value_norm, activation=self.value_activation)  # (, nheads, out_dim // nheads)

            # attention
            graph.apply_edges(fn.e_dot_v(KEY, QUERY, ATTENTION))
            graph.edata[ATTENTION] = self.attn_dropout(edge_softmax(graph, graph.edata[ATTENTION]))
            graph.edata[FEATURE] = graph.edata[VALUE] * graph.edata[ATTENTION]
            graph.update_all(fn.copy_e(FEATURE, FEATURE), fn.sum(FEATURE, FEATURE))

            # linear
            out_feat = graph.ndata[FEATURE].view(graph.num_nodes(), -1)
            out_feat = nodewise_linear(
                out_feat, weight=graph.ndata[self.NODE_WEIGHT], bias=getattr(graph.ndata, self.NODE_BIAS, None),
                norm=self.node_norm, activation=self.node_activation)
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
        return 'num_parameters={}, residual={}, '.format(
            num_parameters(self), self.residual)
