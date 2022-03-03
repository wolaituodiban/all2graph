import all2graph as ag


def test_parse():
    raw_graph = ag.graph_parser.RawGraph()
    raw_graph.add_kv_(0, 'a', 'b')
    raw_graph.add_kv_(0, 'a', 3)
    raw_graph.add_kv_(0, ('a', 'b'), ['haha', 1])
    raw_graph.add_kv_(0, ('a', 'b'), 'c')
    raw_graph.add_kv_(0, 'a', 'b')
    raw_graph.add_kv_(1, 'a', '0.23')
    raw_graph.add_kv_(0, ('a', 'b'), 'b')
    raw_graph.add_kv_(1, ('a', 'b'), 'c')
    raw_graph.add_kv_(2, ('a', 'b'), 'c')
    raw_graph.add_gid_('gkey', 'hehe')
    raw_graph.add_readouts_(['a', 'haha'])
    print(raw_graph)
    meta_info = raw_graph.meta_info()
    graph_parser = ag.GraphParser.from_data(meta_info, add_self_loop=True, to_simple=True)
    graph = graph_parser(raw_graph)
    print(graph)
    print(graph.sample_subgraph(0))
    print(graph.sample_subgraph(1))
    print(graph.sample_subgraph(2))


if __name__ == '__main__':
    test_parse()
