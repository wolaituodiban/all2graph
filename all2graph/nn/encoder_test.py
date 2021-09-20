import torch

from all2graph import Graph, PRESERVED_WORDS, Timer
from all2graph.graph.parser import GraphParser
from all2graph.nn import UGFM, num_parameters


def test():
    print('PRESERVED_WORDS:', PRESERVED_WORDS)
    d_model = 8
    num_latent = 4
    nhead = 2
    num_layers = [(2, 3), (1, 1)]
    transer = GraphParser({}, strings=[], keys=[], targets=['a', 'b'])
    model1 = UGFM(transer, d_model=d_model, num_latent=num_latent, nhead=nhead, num_layers=num_layers, share_conv=True)
    model2 = UGFM(transer, d_model=d_model, num_latent=num_latent, nhead=nhead, num_layers=num_layers, share_conv=False)
    assert num_parameters(model1) < num_parameters(model2)
    print(model1.eval())
    for k, v in model1.named_buffers(recurse=False):
        print(k, v.shape)
        assert (v != 0).all()

    graph = Graph(component_id=[0, 0], key=['key', 'meta'], value=['value', 'node'], type=['readout', 'value'],
                  src=[0, 1, 1], dst=[0, 0, 1])
    with Timer('cpu forward'):
        out = model1(graph)

    if torch.cuda.is_available():
        model1 = model1.cuda()
        with Timer('gpu forward'):
            model1(graph)
        model1.train()
        with Timer('gpu forward'):
            model1(graph)
        with Timer('gpu forward'):
            out = model1(graph)
    print(out)
    assert out.shape == (graph.num_components, transer.num_targets)


if __name__ == '__main__':
    test()
