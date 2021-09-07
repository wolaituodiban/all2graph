import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import torch
import torch.nn.functional as F

from ..globals import FEATURE, KEY, ATTENTION_KEY_WEIGHT, ATTENTION_KEY_BIAS, QUERY, ATTENTION_WEIGHT


class Conv(torch.nn.Module):
    def __init__(self, activation):
        super().__init__()
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        elif isinstance(activation, torch.nn.Module):
            self.activation = activation
        else:
            raise TypeError('activation must be string or pytorch module')

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        with graph.local_scope():
            # feature shape: [num_nodes, embedding_dim]
            graph.ndata[FEATURE] = feat
            graph.apply_edges(fn.copy_u(FEATURE, FEATURE))

            feat = feat.view(graph.num_nodes(), -1, 1)
            # 通过feature计算key
            graph.ndata[KEY] = (feat * graph.ndata[ATTENTION_KEY_WEIGHT]).sum(-1) + graph.ndata[ATTENTION_KEY_BIAS]

            graph.apply_edges(fn.u_dot_e(FEATURE, ATTENTION_KEY_WEIGHT, KEY))
            graph.edata[KEY] = graph.edata[KEY].view(graph.num_edges(), -1)
            graph.edata[KEY] += graph.edata[ATTENTION_KEY_BIAS] + graph.edata[FEATURE]
            graph.apply_edges(fn.e_add_u(KEY, KEY, KEY))

            # # 通过key和query计算attention weight
            graph.apply_edges(fn.e_dot_v(KEY, QUERY, ATTENTION_WEIGHT))
            graph.edata[ATTENTION_WEIGHT] = edge_softmax(graph, graph.edata[ATTENTION_WEIGHT])
            #
            # # 通过feature计算value
            # graph.ndata[globals.VALUE] = (graph.ndata[globals.FEATURE] * graph.ndata[globals.ATTENTION_VALUE_WEIGHT])
            # graph.ndata[globals.VALUE] = graph.ndata[globals.VALUE].sum(-1)
            # graph.ndata[globals.VALUE] += graph.ndata[globals.ATTENTION_VALUE_BIAS]
            # graph.apply_edges(fn.u_dot_e(globals.FEATURE, globals.ATTENTION_VALUE_WEIGHT, globals.VALUE))
            # graph.edata[globals.VALUE] += graph.edata[globals.ATTENTION_VALUE_BIAS] + graph.edata[globals.FEATURE]
            # graph.apply_edges(fn.e_add_u(globals.VALUE, globals.VALUE, globals.VALUE))
            # graph.edata[globals.VALUE] = graph.edata[globals.VALUE].view(graph.num_edges(), self.nheads, -1)
            #
            # # 聚合
            # graph.edata[globals.FEATURE] = graph.edata[globals.FEATURE] * graph.edata[globals.ATTENTION_WEIGHT]
            # graph.edata[globals.FEATURE] = graph.edata[globals.FEATURE].view(graph.num_edges(), 1, -1)
            # graph.update_all(fn.copy_e(globals.FEATURE, globals.FEATURE), fn.sum(globals.FEATURE, globals.FEATURE))
            #
            # # 4 linear
            # output = (graph.ndata[globals.FEATURE] * graph.ndata[globals.FEED_FORWARD_WEIGHT]).sum(-1)
            # output += graph.ndata[globals.FEED_FORWARD_BIAS]
            # output = self.activation(output)
            #
            # # 5 add and norm
            # output = output + graph.ndata[globals.FEATURE].view(output.shape)
            # return output,  graph.edata[globals.FEED_FORWARD_WEIGHT]
