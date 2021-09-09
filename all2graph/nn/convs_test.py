import dgl
import torch
import all2graph as ag
from all2graph.nn import Conv


def test_repr():
    def ffn(x):
        output = x.clone()
        output[:, 0] = 1 / x[:, 0] - 1
        output[:, 1] = x[:, 1] * (output[:, 0] + 1) / output[:, 0]
        return output

    embedding_dim = 2
    nheads = 2
    mid_dim = 1
    dim_per_head = embedding_dim // nheads

    graph = dgl.graph([(0, 0), (1, 0), (2, 0), (1, 1), (2, 2), (2, 1)])
    num_nodes = graph.num_nodes()
    num_edges = graph.num_edges()
    feat = torch.randn(num_nodes, embedding_dim)

    u, v = graph.edges()
    self_loop_mask = u == v

    weight_zeros = torch.zeros(num_edges, 1, nheads, mid_dim, embedding_dim, dtype=torch.float32)
    mid_weight_zeros = torch.zeros(num_edges, 1, nheads, mid_dim, dim_per_head, dtype=torch.float32)
    bias_zeros = torch.zeros(num_edges, 1, nheads, mid_dim, dtype=torch.float32)
    weight_eye = torch.eye(embedding_dim).repeat(num_edges, 1, 1).view(num_edges, 1, nheads, mid_dim, embedding_dim)
    mid_weight_ones = torch.ones(num_edges, 1, nheads, mid_dim, dim_per_head, dtype=torch.float32)
    out_bias_zeros = torch.zeros(num_edges, 1, nheads, dim_per_head, dtype=torch.float32)
    # 手撸参数
    graph.edata[ag.globals.QUERY] = torch.zeros(num_edges, 1, nheads, dim_per_head)

    graph.edata[ag.globals.SRC_KEY_WEIGHT] = weight_zeros
    graph.edata[ag.globals.DST_KEY_WEIGHT] = weight_zeros
    graph.edata[ag.globals.EDGE_KEY_WEIGHT] = mid_weight_zeros
    graph.edata[ag.globals.SRC_KEY_BIAS] = bias_zeros
    graph.edata[ag.globals.DST_KEY_BIAS] = bias_zeros
    graph.edata[ag.globals.EDGE_KEY_BIAS] = out_bias_zeros

    graph.edata[ag.globals.SRC_VALUE_WEIGHT] = -weight_eye.clone()
    graph.edata[ag.globals.SRC_VALUE_WEIGHT][torch.bitwise_not(self_loop_mask), 0, 1] = 0
    graph.edata[ag.globals.DST_VALUE_WEIGHT] = -weight_eye
    graph.edata[ag.globals.EDGE_VALUE_WEIGHT] = mid_weight_ones
    graph.edata[ag.globals.SRC_VALUE_BIAS] = bias_zeros
    graph.edata[ag.globals.DST_VALUE_BIAS] = bias_zeros
    graph.edata[ag.globals.EDGE_VALUE_BIAS] = out_bias_zeros.clone()
    graph.edata[ag.globals.EDGE_VALUE_BIAS][self_loop_mask, 0, 0] = 1

    conv = Conv(None, activation=None).eval()
    print(feat)
    feat, attn = conv(graph, feat, 0)
    print(ffn(feat))
    print(attn.shape)


def test():
    num_nodes = 23
    num_edges = 97
    nheads = 3
    embedding_dim = 12
    feedforward_dim = 13

    # 准备graph
    graph = dgl.graph((torch.randint(num_nodes, (num_edges,)), torch.randint(num_nodes, (num_edges,))))

    graph.edata[ag.QUERY] = torch.randn(num_edges, 1, nheads, embedding_dim // nheads)

    graph.edata[ag.SRC_KEY_WEIGHT] = torch.randn(num_edges, 1, nheads, feedforward_dim, embedding_dim)
    graph.edata[ag.DST_KEY_WEIGHT] = torch.randn(num_edges, 1, nheads, feedforward_dim, embedding_dim)
    graph.edata[ag.SRC_KEY_BIAS] = torch.randn(num_edges, 1, nheads, feedforward_dim)
    graph.edata[ag.DST_KEY_BIAS] = torch.randn(num_edges, 1, nheads, feedforward_dim)
    graph.edata[ag.EDGE_KEY_WEIGHT] = torch.randn(num_edges, 1, nheads, feedforward_dim, embedding_dim // nheads)
    graph.edata[ag.EDGE_KEY_BIAS] = torch.randn(num_edges, 1, nheads, embedding_dim // nheads)

    graph.edata[ag.SRC_VALUE_WEIGHT] = torch.randn(num_edges, 1, nheads, feedforward_dim, embedding_dim)
    graph.edata[ag.DST_VALUE_WEIGHT] = torch.randn(num_edges, 1, nheads, feedforward_dim, embedding_dim)
    graph.edata[ag.SRC_VALUE_BIAS] = torch.randn(num_edges, 1, nheads, feedforward_dim)
    graph.edata[ag.DST_VALUE_BIAS] = torch.randn(num_edges, 1, nheads, feedforward_dim)
    graph.edata[ag.EDGE_VALUE_WEIGHT] = torch.randn(num_edges, 1, nheads, feedforward_dim, embedding_dim // nheads)
    graph.edata[ag.EDGE_VALUE_BIAS] = torch.randn(num_edges, 1, nheads, embedding_dim // nheads)

    # 准备feat
    feat = torch.randn(num_nodes, embedding_dim)

    # run
    conv = Conv(embedding_dim // nheads)
    output, attn = conv(graph, feat, 0)
    assert output.shape == (num_nodes, embedding_dim)
    assert attn.shape == (num_edges, nheads)


if __name__ == '__main__':
    test_repr()
    test()
