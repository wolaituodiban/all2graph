import all2graph as ag


def test_parse():
    raw_graph = ag.graph.RawGraph()
    raw_graph.add_sample_()
    raw_graph.add_kv_('a', 'b')
    raw_graph.add_kv_('a', [1, 2, 3, 4])
    raw_graph.add_kv_('b', 'b')
    raw_graph.add_kv_('b', 'haha')
    raw_graph.add_kv_('c', 'hehe')
    raw_graph.add_kv_('d', 'hihi')
    raw_graph.add_sample_()
    raw_graph.add_kv_('a', 'b')
    raw_graph.add_kv_('a', 0)
    raw_graph.add_kv_('a', 1)
    raw_graph.add_sample_()
    raw_graph.add_kv_('c', 2)
    raw_graph.add_kv_('d', 'haha')
    print(raw_graph)
    meta_info = raw_graph.info()
    graph_parser = ag.GraphParser.from_data(meta_info)
    graph = graph_parser(raw_graph)
    print(graph)


if __name__ == '__main__':
    test_parse()
