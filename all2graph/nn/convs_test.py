import random
import dgl
import torch
import all2graph as ag
from all2graph.nn import Conv


def test_statistics():
    # 实现degree，mean，max三种统计指标
    emb_dim = 3
    nhead = 3

    src_key_weight = torch.tensor(
        [[[[0, 0, 0]],
          [[0, 0, 0]],
          [[0, 0, 1024]]],
         [[[0, 0, 0]],
          [[0, 0, 0]],
          [[0, 0, 0]]]],
        dtype=torch.float32)
    src_key_bias = torch.tensor(
        [[[float('-inf')],
          [0],
          [0]],
         [[1],
          [float('-inf')],
          [float('-inf')]]],
        dtype=torch.float32)

    dst_key_weight = torch.zeros_like(src_key_weight)
    dst_key_bias = torch.zeros_like(src_key_bias)

    src_value_weight = torch.tensor(
        [[[[0, 0, 0]],
          [[0, 1, 0]],
          [[0, 0, 1]]],
         [[[0, 0, 0]],
          [[0, 0, 0]],
          [[0, 0, 0]]]],
        dtype=torch.float32)
    src_value_bias = torch.tensor(
        [[[0],
          [0],
          [0]],
         [[1],
          [0],
          [0]]],
        dtype=torch.float32)
    dst_value_weight = torch.zeros_like(src_value_weight)
    dst_value_bias = torch.zeros_like(src_value_bias)

    query = torch.tensor(
        [[1], [1], [1]],
        dtype=torch.float32)
    node_weight = torch.eye(emb_dim, dtype=torch.float32)
    node_bias = torch.zeros((emb_dim, ))

    num_nodes = 3
    num_edges = 3
    graph = dgl.graph(((0, 1, 2), (0, 0, 0)), num_nodes=num_nodes)
    u, v = graph.edges()
    meta_edge_id = (u == v).long()

    graph.edata[ag.SRC_KEY_WEIGHT] = src_key_weight[meta_edge_id]
    graph.edata[ag.SRC_KEY_BIAS] = src_key_bias[meta_edge_id]
    graph.edata[ag.DST_KEY_WEIGHT] = dst_key_weight[meta_edge_id]
    graph.edata[ag.DST_KEY_BIAS] = dst_key_bias[meta_edge_id]

    graph.edata[ag.SRC_VALUE_WEIGHT] = src_value_weight[meta_edge_id]
    graph.edata[ag.SRC_VALUE_BIAS] = src_value_bias[meta_edge_id]
    graph.edata[ag.DST_VALUE_WEIGHT] = dst_value_weight[meta_edge_id]
    graph.edata[ag.DST_VALUE_BIAS] = dst_value_bias[meta_edge_id]

    graph.ndata[ag.QUERY] = query.repeat(num_nodes, 1, 1)
    graph.ndata[ag.NODE_WEIGHT] = node_weight.repeat(num_nodes, 1, 1)
    graph.ndata[ag.NODE_BIAS] = node_bias.repeat(num_nodes, 1)

    feat = torch.randn(num_nodes, emb_dim)
    conv = Conv(None, activation=None, dropout=0, residual=False)
    out_feat, key_feat, value_feat, attn_weight = conv(graph, feat)
    print(u)
    print(v)
    print(feat)
    print(key_feat)
    print(value_feat)
    print(attn_weight)
    print(out_feat)


def test_shape():
    emb_dim = 16
    nheads = 4
    out_dim = emb_dim
    dim_per_head = out_dim // nheads

    num_nodes = random.randint(0, 30)
    num_edges = random.randint(0, 100)
    graph = dgl.graph(
        (torch.randint(num_nodes-1, (num_edges,)), torch.randint(num_nodes-1, (num_edges,))),
        num_nodes=num_nodes)

    graph.edata[ag.SRC_KEY_WEIGHT] = torch.randn(num_edges, nheads, dim_per_head, emb_dim)
    graph.edata[ag.SRC_KEY_BIAS] = torch.randn(num_edges, nheads, dim_per_head)
    graph.edata[ag.DST_KEY_WEIGHT] = torch.randn(num_edges, nheads, dim_per_head, emb_dim)
    graph.edata[ag.DST_KEY_BIAS] = torch.randn(num_edges, nheads, dim_per_head)

    graph.edata[ag.SRC_VALUE_WEIGHT] = torch.randn(num_edges, nheads, dim_per_head, emb_dim)
    graph.edata[ag.SRC_VALUE_BIAS] = torch.randn(num_edges, nheads, dim_per_head)
    graph.edata[ag.DST_VALUE_WEIGHT] = torch.randn(num_edges, nheads, dim_per_head, emb_dim)
    graph.edata[ag.DST_VALUE_BIAS] = torch.randn(num_edges, nheads, dim_per_head)

    graph.ndata[ag.QUERY] = torch.randn(num_nodes, nheads, dim_per_head)
    graph.ndata[ag.NODE_WEIGHT] = torch.randn(num_nodes, out_dim, emb_dim)
    graph.ndata[ag.NODE_BIAS] = torch.randn(num_nodes, out_dim)

    feat = torch.randn(num_nodes, emb_dim)
    conv = Conv(out_dim)
    out_feat, key_feat, value_feat, attn_weight = conv(graph, feat)
    assert out_feat.shape == (num_nodes, out_dim)
    assert key_feat.shape == (num_edges, out_dim)
    assert value_feat.shape == (num_edges, out_dim)
    assert attn_weight.shape == (num_edges, nheads)


if __name__ == '__main__':
    test_statistics()
    test_shape()
