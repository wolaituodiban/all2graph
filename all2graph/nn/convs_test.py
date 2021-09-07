import dgl
import torch
import all2graph as ag
from all2graph.nn import Conv


def test():
    graph = dgl.graph([(0, 0), (1, 2)])
    graph.ndata[ag.globals.ATTENTION_KEY_WEIGHT] = torch.randn(3, 8, 8)
    graph.ndata[ag.globals.ATTENTION_KEY_BIAS] = torch.randn(3, 8)

    graph.edata[ag.globals.ATTENTION_KEY_WEIGHT] = torch.randn(2, 8, 8)
    graph.edata[ag.globals.ATTENTION_KEY_BIAS] = torch.randn(2, 8)

    graph.ndata['query'] = torch.randn(3, 8)

    feat = torch.randn(3, 8)
    conv = Conv('relu')
    conv(graph, feat)


if __name__ == '__main__':
    test()
