import all2graph as ag


def test_parse():
    raw_graph = ag.graph_parser.RawGraph()
    raw_graph.add_kv_(0, 'a', 'b', True)
    raw_graph.add_kv_(0, 'a', 3, True)
    raw_graph.add_kv_(0, ('a', 'b'), ['haha', 1], False)
    raw_graph.add_kv_(0, ('a', 'b'), 'c', True)
    raw_graph.add_kv_(0, 'a', 'b', True)
    raw_graph.add_kv_(1, 'a', '0.23', True)
    raw_graph.add_kv_(0, ('a', 'b'), 'b', False)
    raw_graph.add_kv_(1, ('a', 'b'), 'c', True)
    raw_graph.add_kv_(2, ('a', 'b'), 'c', True)
    raw_graph.add_readouts_(['a', 'haha'])
    print(raw_graph)
    meta_info = raw_graph.meta_info()
    graph_parser = ag.GraphParser.from_data(meta_info)
    graph = graph_parser(raw_graph)
    print(graph)


if __name__ == '__main__':
    test_parse()
