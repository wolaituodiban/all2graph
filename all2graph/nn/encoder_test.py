import all2graph as ag


def test_load_pretrained():
    # 测试思路：
    # 1、构造两个测试样本s1和s2
    # 2、根据s1构造解析器p1和模型m1
    # 3、根据s1和s2构造解析器p2和模型m2
    # 4、m2从m1加载参数
    # 如果代码正确，那么使用m1和m2计算s1的输出结果应该相同

    # 1、构造两个测试样本s1和s2
    s1 = ag.graph.RawGraph(
        component_id=[0, 0, 0], key=['k1', 'k2', 'k2'], value=['v1', 'v2', 'v5'], src=[1], dst=[0],
        symbol=['readout', 'value', 'value'])
    s2 = ag.graph.RawGraph(
        component_id=[0, 0, 0], key=['k1', 'k2', 'k2'], value=['v3', 'v4', 'v1'], src=[1], dst=[0],
        symbol=['readout', 'value', 'value'])
    s3 = ag.graph.RawGraph.batch([s2, s1])

    # 2、根据s1构造解析器p1和模型m1
    i1 = ag.MetaInfo.from_data(s1)
    p1 = ag.RawGraphParser.from_data(i1, targets=['target'])
    m1 = ag.nn.Encoder(
        p1.num_strings, d_model=8, nhead=2, num_layers=[3], num_weight=False, key_emb=False,
        conv_configs={'key_bias': False, 'value_bias': False, 'node_bias': False}, output_configs={'bias': False}
    )

    # 3、根据s1和s2构造解析器p2和模型m2
    i2 = ag.MetaInfo.from_data(s3)
    p2 = ag.RawGraphParser.from_data(i2, targets=['target'])
    m2 = ag.nn.Encoder(
        p2.num_strings, d_model=8, nhead=2, num_layers=[3], num_weight=False, key_emb=False,
        conv_configs={'key_bias': False, 'value_bias': False, 'node_bias': False}, output_configs={'bias': False}
    )

    # 4、m2从m1加载参数
    m2.load_pretrained(m1, self_parser=p2, other_parser=p1)

    # 如果代码正确，那么使用m1和m2计算s1的输出结果应该相同
    g1 = p1.parse(s1)
    o1 = m1.value_embedding(g1.value)

    g2 = p2.parse(s1)
    o2 = m2.value_embedding(g2.value)

    assert (g1.value != g2.value).any()
    assert (o1 == o2).all()


if __name__ == '__main__':
    test_load_pretrained()
