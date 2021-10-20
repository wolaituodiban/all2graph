import torch
import all2graph as ag
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
        encoder=Encoder(
            num_embeddings=parser.num_strings, d_model=d_model, nhead=nhead, num_layers=[2, 3],
            output_configs={'share_block_param': True}
        ),
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


def test_mock_load_pretrained():
    # 测试思路：
    # 1、构造两个测试样本s1和s2
    # 2、根据s1构造解析器p1和模型m1
    # 3、根据s1和s2构造解析器p2和模型m2
    # 4、m2从m1加载参数
    # 如果代码正确，那么使用m1和m2计算s1的输出结果应该相同

    # 1、构造两个测试样本s1和s2
    s1 = ag.RawGraph(
        component_id=[0, 0, 0, 0, 0], key=['k1', 'k2', 'k2', 'k4', 'k4'], value=['v1', 'v2', 'v5', 0.5, 0.4],
        src=[1], dst=[0], symbol=['readout', 'value', 'value', 'value', 'value'])
    s2 = ag.RawGraph(
        component_id=[0, 0, 0, 0, 0], key=['k1', 'k2', 'k3', 'k4', 'k4'], value=['v3', 'v4', 'v1', -0.5, 0.3],
        src=[1], dst=[0], symbol=['readout', 'value', 'value', 'value', 'value'])
    s3 = ag.RawGraph.batch([s2, s1])

    # 2、根据s1构造解析器p1和模型m1
    i1 = ag.MetaInfo.from_data(s1)
    p1 = ag.RawGraphParser.from_data(i1, targets=['target'])
    m1 = ag.nn.Encoder(p1.num_strings, d_model=8, nhead=2, num_layers=[3, 2])
    m1 = ag.nn.EncoderMetaLearnerMocker(p1, m1).eval()

    # 3、根据s1和s2构造解析器p2和模型m2
    i2 = ag.MetaInfo.from_data(s3)
    p2 = ag.RawGraphParser.from_data(i2, targets=['target'])
    m2 = ag.nn.Encoder(p2.num_strings, d_model=8, nhead=2, num_layers=[3, 2])
    m2 = ag.nn.EncoderMetaLearnerMocker(p2, m2).eval()

    # 4、m2从m1加载参数
    m2.load_pretrained(m1, load_meta_number=True)

    # 如果代码正确，那么使用m1和m2计算s1的输出结果应该相同
    output1, value_info1 = m1(s1, details=True)
    output2, value_info2 = m2(s1, details=True)

    for k in value_info1:
        for i in range(len(value_info1[k])):
            assert (value_info1[k][i] == value_info2[k][i]).all(), (k, i)
    assert output1 == output2


if __name__ == '__main__':
    test_learner()
    test_mock()
    test_mock_load_pretrained()
