import random
import dgl
import torch
from all2graph.nn import HeteroAttnConv


print(HeteroAttnConv.NODE_PARAMS_1D)
print(HeteroAttnConv.NODE_PARAMS_2D)
print(HeteroAttnConv.EDGE_PARAMS_1D)
print(HeteroAttnConv.EDGE_PARAMS_2D)


def test_statistics():
    # 实现degree，mean，max三种统计指标
    emb_dim = 3

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

    graph.edata[HeteroAttnConv.SRC_KEY_WEIGHT] = src_key_weight[meta_edge_id]
    graph.edata[HeteroAttnConv.SRC_KEY_BIAS] = src_key_bias[meta_edge_id]
    graph.edata[HeteroAttnConv.DST_KEY_WEIGHT] = dst_key_weight[meta_edge_id]
    graph.edata[HeteroAttnConv.DST_KEY_BIAS] = dst_key_bias[meta_edge_id]

    graph.edata[HeteroAttnConv.SRC_VALUE_WEIGHT] = src_value_weight[meta_edge_id]
    graph.edata[HeteroAttnConv.SRC_VALUE_BIAS] = src_value_bias[meta_edge_id]
    graph.edata[HeteroAttnConv.DST_VALUE_WEIGHT] = dst_value_weight[meta_edge_id]
    graph.edata[HeteroAttnConv.DST_VALUE_BIAS] = dst_value_bias[meta_edge_id]

    graph.ndata[HeteroAttnConv.QUERY] = query.repeat(num_nodes, 1, 1)
    graph.ndata[HeteroAttnConv.NODE_WEIGHT] = node_weight.repeat(num_nodes, 1, 1)
    graph.ndata[HeteroAttnConv.NODE_BIAS] = node_bias.repeat(num_nodes, 1)

    feat = torch.randn(num_nodes, emb_dim)
    conv = HeteroAttnConv(None, activation=None, dropout=0, residual=False)
    node_feat, edge_feat, attn_weight = conv(graph, feat)
    print(u)
    print(v)
    print(feat)
    print(edge_feat)
    print(attn_weight)
    print(node_feat)


def test_shape():
    emb_dim = 16
    nhead = 4
    out_dim = emb_dim
    dim_per_head = out_dim // nhead

    num_nodes = random.randint(0, 30)
    num_edges = random.randint(0, 100)
    graph = dgl.graph(
        (torch.randint(num_nodes-1, (num_edges,)), torch.randint(num_nodes-1, (num_edges,))),
        num_nodes=num_nodes)

    graph.edata[HeteroAttnConv.SRC_KEY_WEIGHT] = torch.randn(num_edges, nhead, dim_per_head, emb_dim)
    graph.edata[HeteroAttnConv.SRC_KEY_BIAS] = torch.randn(num_edges, nhead, dim_per_head)
    graph.edata[HeteroAttnConv.DST_KEY_WEIGHT] = torch.randn(num_edges, nhead, dim_per_head, emb_dim)
    graph.edata[HeteroAttnConv.DST_KEY_BIAS] = torch.randn(num_edges, nhead, dim_per_head)

    graph.edata[HeteroAttnConv.SRC_VALUE_WEIGHT] = torch.randn(num_edges, nhead, dim_per_head, emb_dim)
    graph.edata[HeteroAttnConv.SRC_VALUE_BIAS] = torch.randn(num_edges, nhead, dim_per_head)
    graph.edata[HeteroAttnConv.DST_VALUE_WEIGHT] = torch.randn(num_edges, nhead, dim_per_head, emb_dim)
    graph.edata[HeteroAttnConv.DST_VALUE_BIAS] = torch.randn(num_edges, nhead, dim_per_head)

    graph.ndata[HeteroAttnConv.QUERY] = torch.randn(num_nodes, nhead, dim_per_head)
    graph.ndata[HeteroAttnConv.NODE_WEIGHT] = torch.randn(num_nodes, out_dim, emb_dim)
    graph.ndata[HeteroAttnConv.NODE_BIAS] = torch.randn(num_nodes, out_dim)

    feat = torch.randn(num_nodes, emb_dim)
    conv = HeteroAttnConv(out_dim)
    node_feat, edge_feat, attn_weight = conv(graph, feat)
    assert node_feat.shape == (num_nodes, out_dim)
    assert edge_feat.shape == (num_edges, out_dim)
    assert attn_weight.shape == (num_edges, nhead)


if __name__ == '__main__':
    test_statistics()
    test_shape()
