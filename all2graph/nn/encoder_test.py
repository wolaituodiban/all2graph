import torch

from all2graph import RawGraph, Timer, MetaInfo
from all2graph.parsers.graph import RawGraphParser
from all2graph.nn import Encoder, MockEncoder


def test_encoder():
    graph = RawGraph(component_id=[0, 0, 0, 0], key=['readout', 'meta haha', 'a', 'a'], value=['a', 'b', 1, 2],
                     symbol=['readout', 'value', 'value', 'value'], src=[0, 1, 1, 2, 3], dst=[0, 0, 1, 0, 0])

    meta_info = MetaInfo.from_data(graph)
    graph_parser = RawGraphParser.from_data(meta_info, targets=['a', 'b'])
    d_model = 8
    nhead = 2
    num_latent = 4
    model1 = Encoder(
        graph_parser, d_model=d_model, nhead=nhead, num_latent=num_latent, num_layers=[2, 2], num_meta_layers=[2, 2])
    print(model1.eval())
    with Timer('cpu forward'):
        out = model1(graph, details=True)

    if torch.cuda.is_available():
        model1 = model1.cuda()
        with Timer('gpu forward'):
            model1(graph, details=True)
        with Timer('gpu forward'):
            out = model1(graph, details=True)
    out[0]['a'].mean().backward()
    assert len(out[0]) == graph_parser.num_targets
    for v in out[0].values():
        assert v.shape == (graph.num_components, ), (v.shape, graph.num_components)


def test_mock():
    graph = RawGraph(component_id=[0, 0, 0, 0], key=['readout', 'meta haha', 'a', 'a'], value=['a', 'b', 1, 2],
                     symbol=['readout', 'value', 'value', 'value'], src=[0, 1, 1, 2, 3], dst=[0, 0, 1, 0, 0])

    meta_info = MetaInfo.from_data(graph)
    graph_parser = RawGraphParser.from_data(meta_info, targets=['a', 'b'])
    d_model = 8
    nhead = 2
    model1 = MockEncoder(
        graph_parser, d_model=d_model, nhead=nhead, num_layers=[2, 2]
    )
    print(model1.eval())
    with Timer('cpu forward'):
        out = model1(graph, details=True)

    if torch.cuda.is_available():
        model1 = model1.cuda()
        with Timer('gpu forward'):
            model1(graph, details=True)
        with Timer('gpu forward'):
            out = model1(graph, details=True)
    out[0]['a'].mean().backward()
    assert len(out[0]) == graph_parser.num_targets
    for v in out[0].values():
        assert v.shape == (graph.num_components, ), (v.shape, graph.num_components)


if __name__ == '__main__':
    test_encoder()
    test_mock()
