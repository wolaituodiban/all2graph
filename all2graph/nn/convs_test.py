import dgl
import torch
import all2graph as ag
from all2graph.nn import Conv


def test_count():
    ffn = lambda x: 1 / x - 1
    feat = torch.randn(3, 1)
    graph = dgl.graph([(0, 0), (1, 0), (2, 0), (1, 1), (2, 2)])
    num_nodes = graph.num_nodes()
    num_edges = graph.num_edges()

    graph.ndata[ag.globals.QUERY] = torch.zeros(num_nodes, 1)
    graph.edata[ag.globals.SRC_KEY_BIAS] = torch.zeros(num_edges, 1, dtype=torch.float32)
    graph.edata[ag.globals.SRC_KEY_WEIGHT] = torch.zeros(num_edges, 1, 1, dtype=torch.float32)
    graph.edata[ag.globals.DST_KEY_BIAS] = torch.zeros(num_edges, 1, dtype=torch.float32)
    graph.edata[ag.globals.DST_KEY_WEIGHT] = torch.zeros(num_edges, 1, 1, dtype=torch.float32)
    graph.edata[ag.globals.EDGE_KEY_BIAS] = torch.zeros(num_edges, 1, dtype=torch.float32)
    graph.edata[ag.globals.EDGE_KEY_WEIGHT] = torch.zeros(num_edges, 1, 1, dtype=torch.float32)

    graph.edata[ag.globals.SRC_VALUE_BIAS] = torch.zeros(num_edges, 1, dtype=torch.float32)
    graph.edata[ag.globals.SRC_VALUE_WEIGHT] = - torch.ones(num_edges, 1, 1, dtype=torch.float32)
    graph.edata[ag.globals.DST_VALUE_BIAS] = torch.zeros(num_edges, 1, dtype=torch.float32)
    graph.edata[ag.globals.DST_VALUE_WEIGHT] = - torch.ones(num_edges, 1, 1, dtype=torch.float32)
    graph.edata[ag.globals.EDGE_VALUE_BIAS] = torch.tensor([1, 0, 0, 1, 1], dtype=torch.float32).view(-1, 1)
    graph.edata[ag.globals.EDGE_VALUE_WEIGHT] = torch.ones(num_edges, 1, 1, dtype=torch.float32)

    conv = Conv(1, activation=None, norm=False).eval()
    feat, attn = conv(graph, feat)
    print(ffn(feat))


if __name__ == '__main__':
    test_count()
