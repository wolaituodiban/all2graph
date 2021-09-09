import dgl
import torch
import all2graph as ag
from all2graph.nn import Readout


def test():
    num_nodes = 97
    num_layers = 5
    nheads = 3
    embedding_dim = 12
    feedforward_dim = 13

    # 准备graph
    graph = dgl.graph(([0], [0]), num_nodes=num_nodes)
    graph.ndata[ag.COMPONENT_ID] = torch.randint(0, 7, (num_nodes,))
    num_components = graph.ndata[ag.COMPONENT_ID].unique().shape[0]

    graph.ndata[ag.QUERY] = torch.randn(num_nodes, num_layers, nheads, embedding_dim // nheads)

    graph.ndata[ag.NODE_KEY_WEIGHT_1] = torch.randn(num_nodes, num_layers, nheads, feedforward_dim, embedding_dim)
    graph.ndata[ag.NODE_KEY_BIAS_1] = torch.randn(num_nodes, num_layers, nheads, feedforward_dim)
    graph.ndata[ag.NODE_KEY_WEIGHT_2] = torch.randn(
        num_nodes, num_layers, nheads, feedforward_dim, embedding_dim // nheads
    )
    graph.ndata[ag.NODE_KEY_BIAS_2] = torch.randn(num_nodes, num_layers, nheads, embedding_dim // nheads)

    graph.ndata[ag.NODE_VALUE_WEIGHT_1] = torch.randn(num_nodes, num_layers, nheads, feedforward_dim, embedding_dim)
    graph.ndata[ag.NODE_VALUE_BIAS_1] = torch.randn(num_nodes, num_layers, nheads, feedforward_dim)
    graph.ndata[ag.NODE_VALUE_WEIGHT_2] = torch.randn(
        num_nodes, num_layers, nheads, feedforward_dim, embedding_dim // nheads
    )
    graph.ndata[ag.NODE_VALUE_BIAS_2] = torch.randn(num_nodes, num_layers, nheads, embedding_dim // nheads)

    # 准备feat
    feat = torch.randn(num_nodes, num_layers, embedding_dim)

    # run
    readout = Readout(embedding_dim // nheads)
    output, attn, component_id = readout(graph, feat)
    assert output.shape == (num_components, embedding_dim)
    assert attn.shape == (num_components, num_nodes, num_layers, nheads)
    assert (component_id == graph.ndata[ag.COMPONENT_ID].unique()).all()
    assert (attn != 0).any(-1).any(-1).sum() == num_nodes


if __name__ == '__main__':
    test()
