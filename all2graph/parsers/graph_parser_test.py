import all2graph as ag
import torch


def test_parse():
    raw_graph = ag.graph.RawGraph()
    raw_graph.add_kv_('a', 'b')
    raw_graph.add_kv_('a', [1, 2, 3, 4])
    raw_graph.add_kv_('b', 'b')
    raw_graph.add_kv_('b', 'haha')
    raw_graph.add_kv_('c', 'hehe')
    raw_graph.add_kv_('d', 'hihi')
    raw_graph.add_split_()
    raw_graph.add_kv_('a', 'b')
    raw_graph.add_kv_('a', 0)
    raw_graph.add_kv_('a', 1)
    raw_graph.add_split_()
    raw_graph.add_kv_('c', 2)
    raw_graph.add_kv_('d', 'haha')
    raw_graph.add_split_()
    raw_graph._assert()
    print(raw_graph.nodes_per_sample)
    print(raw_graph)
    meta_info = ag.MetaInfo.from_data(raw_graph)
    graph_parser = ag.GraphParser.from_data(meta_info)
    graph = graph_parser(raw_graph)
    print(graph)
    print(graph.add_self_loop())
    print(graph.to_bidirectied(copy_ndata=True))
    print(graph.to_simple(copy_ndata=True))
    for ids in graph.seq_ids:
        assert torch.unique(graph.keys[ids]).shape[0]


if __name__ == '__main__':
    test_parse()
