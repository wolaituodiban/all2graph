import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import torch

from .functional import edgewise_feedforward
from ..globals import FEATURE, KEY, VALUE, ATTENTION, QUERY, \
    SRC_KEY_BIAS, SRC_KEY_WEIGHT, DST_KEY_BIAS, DST_KEY_WEIGHT, EDGE_KEY_BIAS, EDGE_KEY_WEIGHT, \
    SRC_VALUE_BIAS, SRC_VALUE_WEIGHT, DST_VALUE_BIAS, DST_VALUE_WEIGHT, EDGE_VALUE_BIAS, EDGE_VALUE_WEIGHT


class Conv(torch.nn.Module):
    def __init__(self, normalized_shape, activation='relu', dropout=0.1):
        super().__init__()
        if activation == 'gelu':
            activation = torch.nn.GELU()
        elif activation == 'relu':
            activation = torch.nn.ReLU()
        self.key_activation = activation
        self.value_activation = activation

        self.key_dropout1 = torch.nn.Dropout(dropout)
        self.key_dropout2 = torch.nn.Dropout(dropout)
        self.value_dropout1 = torch.nn.Dropout(dropout)
        self.value_dropout2 = torch.nn.Dropout(dropout)
        self.attn_dropout = torch.nn.Dropout(dropout)

        if normalized_shape is not None:
            self.key_norm = torch.nn.LayerNorm(normalized_shape)
            self.value_norm = torch.nn.LayerNorm(normalized_shape)
            self.out_norm = torch.nn.LayerNorm(normalized_shape)
        else:
            self.key_norm = None
            self.value_norm = None
            self.out_norm = None

    def reset_parameters(self):
        self.key_norm.reset_parameters()
        self.value_norm.reset_parameters()

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor, i: int) -> (torch.Tensor, torch.Tensor):
        """
        :param graph:
            edata:
                QUERY            : (, num_layers, nheads, embedding_dim // nheads)

                SRC_KEY_WEIGHT   : (, num_layers, nheads, feedforward_dim, embedding_dim)
                DST_KEY_WEIGHT   : (, num_layers, nheads, feedforward_dim, embedding_dim)
                SRC_KEY_BIAS     : (, num_layers, nheads, feedforward_dim)
                DST_KEY_BIAS     : (, num_layers, nheads, feedforward_dim)
                EDGE_KEY_WEIGHT  : (, num_layers, nheads, feedforward_dim, embedding_dim // nheads)
                EDGE_KEY_BIAS    : (, num_layers, nheads, embedding_dim // nheads)

                SRC_VALUE_WEIGHT : (, num_layers, nheads, feedforward_dim, embedding_dim)
                DST_VALUE_WEIGHT : (, num_layers, nheads, feedforward_dim, embedding_dim)
                SRC_VALUE_BIAS   : (, num_layers, nheads, feedforward_dim)
                DST_VALUE_BIAS   : (, num_layers, nheads, feedforward_dim)
                EDGE_VALUE_WEIGHT: (, num_layers, nheads, feedforward_dim, embedding_dim // nheads)
                EDGE_VALUE_BIAS  : (, num_layers, nheads, embedding_dim // nheads)
        :param feat: num_nodes * embedding_dim
        :param i: 第几层，所有参数会按照[:, i]的方式取
        :return:
            FEATURE  : (num_nodes, embedding_dim)
            ATTENTION: (num_edges, nheads)
        """
        with graph.local_scope():
            # 通过feature计算key
            graph.edata[KEY] = edgewise_feedforward(
                feat=feat, graph=graph, u_weight=graph.edata[SRC_KEY_WEIGHT][:, i],
                v_weight=graph.edata[DST_KEY_WEIGHT][:, i],
                e_weight=graph.edata[EDGE_KEY_WEIGHT][:, i], u_bias=graph.edata[SRC_KEY_BIAS][:, i],
                v_bias=graph.edata[DST_KEY_BIAS][:, i], e_bias=graph.edata[EDGE_KEY_BIAS][:, i],
                activation=self.key_activation, dropout1=self.key_dropout1, dropout2=self.key_dropout2,
                norm=self.key_norm
            )

            # # 通过key和query计算attention weight
            attention = (graph.edata[KEY] * graph.edata[QUERY][:, i]).sum(-1, keepdim=True)
            graph.edata[ATTENTION] = self.attn_dropout(edge_softmax(graph, attention))

            # 通过feature计算value
            graph.edata[VALUE] = edgewise_feedforward(
                feat=feat, graph=graph, u_weight=graph.edata[SRC_VALUE_WEIGHT][:, i],
                v_weight=graph.edata[DST_VALUE_WEIGHT][:, i],
                e_weight=graph.edata[EDGE_VALUE_WEIGHT][:, i], u_bias=graph.edata[SRC_VALUE_BIAS][:, i],
                v_bias=graph.edata[DST_VALUE_BIAS][:, i], e_bias=graph.edata[EDGE_VALUE_BIAS][:, i],
                activation=self.value_activation, dropout1=self.value_dropout1, dropout2=self.value_dropout2,
                norm=self.value_norm
            )

            # 加权平均
            graph.edata[FEATURE] = graph.edata[VALUE] * graph.edata[ATTENTION]
            graph.update_all(fn.copy_e(FEATURE, FEATURE), fn.sum(FEATURE, FEATURE))
            if self.out_norm is not None:
                graph.ndata[FEATURE] = self.out_norm(graph.ndata[FEATURE])
            return graph.ndata[FEATURE].view(feat.shape),  graph.edata[ATTENTION].view(attention.shape[:-1])
