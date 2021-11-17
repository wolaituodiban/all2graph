import random
import dgl
import torch
import all2graph as ag
from all2graph.nn import Conv, Block


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
    node_weight = torch.eye(emb_dim, dtype=torch.float32).unsqueeze(0)
    node_bias = torch.zeros((1, emb_dim))

    num_nodes = 3
    num_edges = 3
    graph = dgl.graph(((0, 1, 2), (0, 0, 0)), num_nodes=num_nodes)
    u, v = graph.edges()
    meta_node_id = torch.zeros(graph.num_nodes(), dtype=torch.long)
    meta_edge_id = (u == v).long()

    feat = torch.randn(num_nodes, emb_dim)
    conv = Conv(
        None, node_activation=None, dropout=0, residual=False, node_norm=False, key_norm=False, value_norm=False,
        norm=False)
    out_feat, key, value, attn_weight = conv(
        graph, feat, dict(
            src_key_weight=src_key_weight[meta_edge_id], src_key_bias=src_key_bias[meta_edge_id],
            dst_key_weight=dst_key_weight[meta_edge_id], dst_key_bias=dst_key_bias[meta_edge_id],
            src_value_weight=src_value_weight[meta_edge_id], src_value_bias=src_value_bias[meta_edge_id],
            dst_value_weight=dst_value_weight[meta_edge_id], dst_value_bias=dst_value_bias[meta_edge_id],
            query=query[meta_node_id], node_weight=node_weight[meta_node_id], node_bias=node_bias[meta_node_id]),
    )
    print(u)
    print(v)
    print(feat)
    print(value)
    print(attn_weight)
    print(out_feat)
    assert out_feat[0, 0] == 1
    assert out_feat[0, 1] == feat[1:, 1].mean()
    assert out_feat[0, 2] == feat[1:, 2].max()


def test_shape():
    emb_dim = 16
    nhead = 4
    out_dim = emb_dim
    dim_per_head = out_dim // nhead

    num_nodes = random.randint(1, 3000)
    num_edges = random.randint(1, 10000)
    graph = dgl.graph(
        (torch.randint(num_nodes-1, (num_edges,)), torch.randint(num_nodes-1, (num_edges,))),
        num_nodes=num_nodes)

    src_key_weight = torch.randn(nhead, dim_per_head, emb_dim).expand(num_edges, -1, -1, -1)
    src_key_bias = torch.randn(nhead, dim_per_head).expand(num_edges, -1, -1)
    dst_key_weight = torch.randn(nhead, dim_per_head, emb_dim).expand(num_edges, -1, -1, -1)
    dst_key_bias = torch.randn(nhead, dim_per_head).expand(num_edges, -1, -1)

    src_value_weight = torch.randn(nhead, dim_per_head, emb_dim).expand(num_edges, -1, -1, -1)
    src_value_bias = torch.randn(nhead, dim_per_head).expand(num_edges, -1, -1)
    dst_value_weight = torch.randn(nhead, dim_per_head, emb_dim).expand(num_edges, -1, -1, -1)
    dst_value_bias = torch.randn(nhead, dim_per_head).expand(num_edges, -1, -1)

    query = torch.randn(nhead, dim_per_head).expand(num_nodes, -1, -1)
    node_weight = torch.randn(nhead, dim_per_head, emb_dim).expand(num_nodes, -1, -1, -1)
    node_bias = torch.randn(nhead, dim_per_head).expand(num_nodes, -1, -1)

    feat = torch.randn(num_nodes, emb_dim)
    conv = Conv(out_dim)
    node_feat, key, edge_feat, attn_weight = conv(
        graph, feat, dict(
            src_key_weight=src_key_weight,
            src_key_bias=src_key_bias,
            dst_key_weight=dst_key_weight,
            dst_key_bias=dst_key_bias,
            src_value_weight=src_value_weight,
            src_value_bias=src_value_bias,
            dst_value_weight=dst_value_weight,
            dst_value_bias=dst_value_bias,
            query=query,
            node_weight=node_weight,
            node_bias=node_bias)
    )
    assert node_feat.shape == (num_nodes, out_dim)
    assert key.shape == (num_edges, out_dim)
    assert edge_feat.shape == (num_edges, out_dim)
    assert attn_weight.shape == (num_edges, nhead)

    block = Block(conv, num_layers=2, share_layer=False)
    node_feats, keys, edge_feats, attn_weights = block(
        graph, feat, dict(
            src_key_weight=src_key_weight, src_key_bias=src_key_bias, dst_key_weight=dst_key_weight,
            dst_key_bias=dst_key_bias, src_value_weight=src_value_weight, src_value_bias=src_value_bias,
            dst_value_weight=dst_value_weight, dst_value_bias=dst_value_bias, query=query, node_weight=node_weight,
            node_bias=node_bias)
    )
    for node_feat in node_feats:
        assert node_feat.shape == (num_nodes, out_dim)
    for key in keys:
        assert key.shape == (num_edges, out_dim)
    for edge_feat in edge_feats:
        assert edge_feat.shape == (num_edges, out_dim)
    for attn_weight in attn_weights:
        assert attn_weight.shape == (num_edges, nhead)

    conv.use_matmul = False
    with ag.Timer('*'):
        for _ in range(1000):
            conv(
                graph, feat, dict(
                    src_key_weight=src_key_weight, src_key_bias=src_key_bias, dst_key_weight=dst_key_weight,
                    dst_key_bias=dst_key_bias, src_value_weight=src_value_weight, src_value_bias=src_value_bias,
                    dst_value_weight=dst_value_weight, dst_value_bias=dst_value_bias, query=query,
                    node_weight=node_weight,
                    node_bias=node_bias)
            )

    conv.use_matmul = True
    with ag.Timer('*'):
        for _ in range(1000):
            conv(
                graph, feat, dict(
                    src_key_weight=src_key_weight, src_key_bias=src_key_bias, dst_key_weight=dst_key_weight,
                    dst_key_bias=dst_key_bias, src_value_weight=src_value_weight, src_value_bias=src_value_bias,
                    dst_value_weight=dst_value_weight, dst_value_bias=dst_value_bias, query=query,
                    node_weight=node_weight,
                    node_bias=node_bias)
            )


if __name__ == '__main__':
    test_statistics()
    test_shape()
