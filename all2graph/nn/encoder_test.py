import torch.nn

from all2graph import Graph, PRESERVED_WORDS
from all2graph.graph.transer import GraphTranser
from all2graph.nn import UGFM, num_parameters


def test():
    print('PRESERVED_WORDS:', PRESERVED_WORDS)
    d_model = 8
    num_latent = 4
    nhead = 2
    num_layers = [(2, 3), (1, 1)]
    transer = GraphTranser({}, strings=[], keys=[])
    model1 = UGFM(transer, d_model=d_model, num_latent=num_latent, nhead=nhead, num_layers=num_layers, share_conv=True)
    model2 = UGFM(transer, d_model=d_model, num_latent=num_latent, nhead=nhead, num_layers=num_layers, share_conv=False)
    assert num_parameters(model1) < num_parameters(model2)
    print(model1.eval())
    for k, v in model1._meta_node_param.items():
        assert (v != 0).all()
    for k, v in model1._meta_edge_param.items():
        assert (v != 0).all(), k
    graph = Graph(component_id=[0, 0], key=['key', 'meta'], value=['value', 'node'], type=['value', 'value'],
                  src=[0, 1, 1], dst=[0, 0, 1])
    out = model1(graph)
    for x in out:
        assert x.shape == (graph.num_nodes, d_model)


if __name__ == '__main__':
    test()
