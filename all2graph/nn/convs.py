import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import torch

from .functional import edge_feed_forward
from ..globals import FEATURE, KEY, VALUE, ATTENTION, QUERY, \
    SRC_KEY_BIAS, SRC_KEY_WEIGHT, DST_KEY_BIAS, DST_KEY_WEIGHT, EDGE_KEY_BIAS, EDGE_KEY_WEIGHT, \
    SRC_VALUE_BIAS, SRC_VALUE_WEIGHT, DST_VALUE_BIAS, DST_VALUE_WEIGHT, EDGE_VALUE_BIAS, EDGE_VALUE_WEIGHT


class Conv(torch.nn.Module):
    def __init__(self, embedding_dim, activation='relu', dropout=0.1, norm=True):
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

        if norm:
            self.key_norm = torch.nn.LayerNorm(embedding_dim)
            self.value_norm = torch.nn.LayerNorm(embedding_dim)
            self.attn_norm = torch.nn.LayerNorm(embedding_dim)
        else:
            self.key_norm = None
            self.value_norm = None
            self.attn_norm = None

    def reset_parameters(self):
        self.key_norm.reset_parameters()
        self.value_norm.reset_parameters()

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """

        weights:
            QUERY            :
            SRC_KEY_BIAS     :
            SRC_KEY_WEIGHT   :
            DST_KEY_BIAS     :
            DST_KEY_WEIGHT   :
            EDGE_KEY_BIAS    :
            EDGE_KEY_WEIGHT  :
            SRC_VALUE_BIAS   :
            SRC_VALUE_WEIGHT :
            DST_VALUE_BIAS   :
            DST_VALUE_WEIGHT :
            EDGE_VALUE_BIAS  :
            EDGE_VALUE_WEIGHT:
        intermedia:
            KEY  :
            VALUE:
        outputs:
            FEATURE  :
            ATTENTION:
        """
        with graph.local_scope():
            # 通过feature计算key
            graph.edata[KEY] = edge_feed_forward(
                feat=feat, graph=graph, u_weight=graph.edata[SRC_KEY_WEIGHT], v_weight=graph.edata[DST_KEY_WEIGHT],
                e_weight=graph.edata[EDGE_KEY_WEIGHT], u_bias=graph.edata[SRC_KEY_BIAS],
                v_bias=graph.edata[DST_KEY_BIAS], e_bias=graph.edata[EDGE_KEY_BIAS], activation=self.key_activation,
                dropout1=self.key_dropout1, dropout2=self.key_dropout2, norm=self.key_norm
            )

            # # 通过key和query计算attention weight
            graph.apply_edges(fn.e_dot_v(KEY, QUERY, ATTENTION))
            graph.edata[ATTENTION] = self.attn_dropout(edge_softmax(graph, graph.edata[ATTENTION]))

            # 通过feature计算value
            graph.edata[VALUE] = edge_feed_forward(
                feat=feat, graph=graph, u_weight=graph.edata[SRC_VALUE_WEIGHT], v_weight=graph.edata[DST_VALUE_WEIGHT],
                e_weight=graph.edata[EDGE_VALUE_WEIGHT], u_bias=graph.edata[SRC_VALUE_BIAS],
                v_bias=graph.edata[DST_VALUE_BIAS], e_bias=graph.edata[EDGE_VALUE_BIAS],
                activation=self.value_activation, dropout1=self.value_dropout1, dropout2=self.value_dropout2,
                norm=self.value_norm
            )

            # 加权平均
            graph.edata[FEATURE] = graph.edata[VALUE] * graph.edata[ATTENTION]
            graph.update_all(fn.copy_e(FEATURE, FEATURE), fn.sum(FEATURE, FEATURE))
            if self.attn_norm is not None:
                graph.ndata[FEATURE] = self.attn_norm(graph.ndata[FEATURE])
            return graph.ndata[FEATURE].view(feat.shape),  graph.edata[ATTENTION]
