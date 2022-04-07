import all2graph as ag


def test_from_data():
    graph = ag.graph.RawGraph()
    graph.add_kv_('a', 'b')
    graph.add_kv_('a', [1, 2, 3, 4])
    graph.add_kv_('b', 'b')
    graph.add_kv_('b', 'haha')
    graph.add_kv_('c', 'hehe')
    graph.add_kv_('d', 'hihi')
    graph.add_split_()
    graph_info = ag.MetaInfo.from_data(graph)
    print(graph_info)


def test_reduce():
    graph1 = ag.graph.RawGraph()
    graph1.add_kv_('a', 'b')
    graph1.add_kv_('a', [1, 2, 3, 4])
    graph1.add_kv_('b', 'b')
    graph1.add_kv_('b', 'haha')
    graph1.add_kv_('c', 'hehe')
    graph1.add_kv_('d', 'hihi')
    graph1.add_split_()
    meta_info1 = ag.MetaInfo.from_data(graph1)

    graph2 = ag.graph.RawGraph()
    graph2.add_kv_('a', 'b')
    graph2.add_kv_('a', 0)
    graph2.add_kv_('a', 1)
    graph2.add_split_()
    graph2.add_kv_('c', 2)
    graph2.add_kv_('d', 'haha')
    graph2.add_split_()
    meta_info2 = ag.MetaInfo.from_data(graph2)

    graph3 = ag.graph.RawGraph()
    graph3.add_kv_('a', 'b')
    graph3.add_kv_('a', [1, 2, 3, 4])
    graph3.add_kv_('b', 'b')
    graph3.add_kv_('b', 'haha')
    graph3.add_kv_('c', 'hehe')
    graph3.add_kv_('d', 'hihi')
    graph3.add_split_()
    graph3.add_kv_('a', 'b')
    graph3.add_kv_('a', 0)
    graph3.add_kv_('a', 1)
    graph3.add_split_()
    graph3.add_kv_('c', 2)
    graph3.add_kv_('d', 'haha')
    graph3.add_split_()
    meta_info3 = ag.MetaInfo.from_data(graph3)
    print(meta_info3.num_counts)
    meta_info4 = ag.MetaInfo.batch([meta_info1, meta_info2])
    print(meta_info4.num_counts)
    assert ag.equal(meta_info3, meta_info4)

    print(meta_info4.dictionary())
    print(meta_info4.num_ecdfs)


if __name__ == '__main__':
    test_from_data()
    test_reduce()
