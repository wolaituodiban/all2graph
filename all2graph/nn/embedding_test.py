import random
import numpy as np
import dgl
import torch
from all2graph import VALUE, NUMBER
from all2graph.nn import ValueEmbedding


def test():
    emb_dim = random.randint(1, 7)

    num_embbeddings = random.randint(1, 10)
    num_nodes = random.randint(2, 30)
    num_edges = random.randint(2, 100)
    graph = dgl.graph(
        (torch.randint(num_nodes - 1, (num_edges,)), torch.randint(num_nodes - 1, (num_edges,))),
        num_nodes=num_nodes)
    graph.ndata[VALUE] = torch.randint(num_embbeddings, (num_nodes,))
    graph.ndata[NUMBER] = torch.randn((num_nodes, ))
    graph.ndata[NUMBER].masked_fill_(torch.rand_like(graph.ndata[NUMBER]) < 0.5, np.nan)

    emb = ValueEmbedding(embedding_dim=emb_dim, num_embeddings=num_embbeddings)
    out = emb(graph)
    out.sum().backward()
    assert emb.embedding.weight.grad.max() > 0
    assert out.shape == (num_nodes, emb_dim)


if __name__ == '__main__':
    test()
