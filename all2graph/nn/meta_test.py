import torch

from all2graph import RawGraph, Timer, MetaInfo, RawGraphParser
from all2graph.nn import Encoder, EncoderMetaLearner, EncoderMetaLearnerMocker


def test_learner():
    graph = RawGraph(component_id=[0, 0, 0, 0], key=['readout', 'meta haha', 'a', 'a'], value=['a', 'b', 1, 2],
                     symbol=['readout', 'value', 'value', 'value'], src=[0, 1, 1, 2, 3], dst=[0, 0, 1, 0, 0])

    meta_info = MetaInfo.from_data(graph)
    parser = RawGraphParser.from_data(meta_info, targets=['a', 'b'])
    d_model = 8
    nhead = 2
    num_latent = 4
    model = EncoderMetaLearner(
        raw_graph_parser=parser,
        encoder=Encoder(num_embeddings=parser.num_strings, d_model=d_model, nhead=nhead, num_layers=[2, 3]),
        num_latent=num_latent)
    print(model.eval())
    with Timer('cpu forward'):
        out = model(graph, details=True)

    if torch.cuda.is_available():
        model = model.cuda()
        with Timer('gpu forward'):
            model(graph, details=True)
        with Timer('gpu forward'):
            out = model(graph, details=True)
    out[0]['a'].mean().backward()
    assert len(out[0]) == parser.num_targets
    for v in out[0].values():
        assert v.shape == (graph.num_components, ), (v.shape, graph.num_components)


def test_mock():
    graph = RawGraph(component_id=[0, 0, 0, 0], key=['readout', 'meta haha', 'a', 'a'], value=['a', 'b', 1, 2],
                     symbol=['readout', 'value', 'value', 'value'], src=[0, 1, 1, 2, 3], dst=[0, 0, 1, 0, 0])

    meta_info = MetaInfo.from_data(graph)
    parser = RawGraphParser.from_data(meta_info, targets=['a', 'b'])
    d_model = 8
    nhead = 2
    model = EncoderMetaLearnerMocker(
        raw_graph_parser=parser,
        encoder=Encoder(num_embeddings=parser.num_strings, d_model=d_model, nhead=nhead, num_layers=[2, 3]))
    print(model.eval())
    with Timer('cpu forward'):
        out = model(graph, details=True)

    if torch.cuda.is_available():
        model = model.cuda()
        with Timer('gpu forward'):
            model(graph, details=True)
        with Timer('gpu forward'):
            out = model(graph, details=True)
    out[0]['a'].mean().backward()
    assert len(out[0]) == parser.num_targets
    for v in out[0].values():
        assert v.shape == (graph.num_components, ), (v.shape, graph.num_components)

    # torch.save(model1.state_dict(), 'state_dict.torch')
    # torch.save(out, 'out.torch')


if __name__ == '__main__':
    test_learner()
    test_mock()
