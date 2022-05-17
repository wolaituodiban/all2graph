import jieba

import all2graph as ag
import torch


def test_parse():
    raw_graph = ag.graph.RawGraph()
    raw_graph.add_kv_(0, ag.READOUT, 'b')
    raw_graph.add_kv_(0, 'a', [1, 2, 3, 4])
    raw_graph.add_kv_(0, 'b', 'b')
    raw_graph.add_kv_(0, 'b', 'haha')
    raw_graph.add_kv_(0, 'c', 'hehe')
    raw_graph.add_kv_(0, 'd', 'hihi')
    raw_graph.add_kv_(1, 'a', 'b')
    raw_graph.add_kv_(1, 'a', 0)
    raw_graph.add_kv_(1, 'a', 1)
    raw_graph.add_kv_(2, 'c', 2)
    raw_graph.add_kv_(2, 'dar Tank', 'haha')
    raw_graph._assert()
    print(raw_graph)
    meta_info = ag.MetaInfo.from_data(raw_graph)
    graph_parser = ag.GraphParser.from_data(meta_info)
    graph = graph_parser.call(raw_graph)
    print(graph)
    assert graph.num_nodes == raw_graph.num_nodes
    assert torch.equal(graph.node2seq[graph.seq2node()].flatten(), torch.arange(graph.num_nodes))
    print(graph.add_self_loop())
    print(graph.to_bidirectied(copy_ndata=True))
    print(graph.to_simple(copy_ndata=True))
    print(graph.seq_mask('a'))
    for t, nodes in raw_graph.seq_info().type2node.items():
        assert (graph.types[nodes] == graph.type_mapper[t]).all()


if __name__ == '__main__':
    test_parse()
