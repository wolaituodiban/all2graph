# import all2graph as ag
#
#
# def test_from_data():
#     graph = ag.graph.RawGraph()
#     graph.add_split_()
#     graph.add_kv_('a', 'b')
#     graph.add_kv_('a', [1, 2, 3, 4])
#     graph.add_kv_('b', 'b')
#     graph.add_kv_('b', 'haha')
#     graph.add_kv_('c', 'hehe')
#     graph.add_kv_('d', 'hihi')
#     graph_info = ag.GraphInfo.from_data(indices=graph.indices, values=graph.formatted_values)
#     print(graph_info)
#
#
# def test_reduce():
#     graph1 = ag.graph.RawGraph()
#     graph1.add_split_()
#     graph1.add_kv_('a', 'b')
#     graph1.add_kv_('a', [1, 2, 3, 4])
#     graph1.add_kv_('b', 'b')
#     graph1.add_kv_('b', 'haha')
#     graph1.add_kv_('c', 'hehe')
#     graph1.add_kv_('d', 'hihi')
#     graph_info1 = ag.GraphInfo.from_data(indices=graph1.indices, values=graph1.formatted_values)
#
#     graph2 = ag.graph.RawGraph()
#     graph2.add_split_()
#     graph2.add_kv_('a', 'b')
#     graph2.add_kv_('a', 0)
#     graph2.add_kv_('a', 1)
#     graph2.add_split_()
#     graph2.add_kv_('c', 2)
#     graph2.add_kv_('d', 'haha')
#     graph_info2 = ag.GraphInfo.from_data(indices=graph2.indices, values=graph2.formatted_values)
#
#     graph3 = ag.graph.RawGraph()
#     graph3.add_split_()
#     graph3.add_kv_('a', 'b')
#     graph3.add_kv_('a', [1, 2, 3, 4])
#     graph3.add_kv_('b', 'b')
#     graph3.add_kv_('b', 'haha')
#     graph3.add_kv_('c', 'hehe')
#     graph3.add_kv_('d', 'hihi')
#     graph3.add_split_()
#     graph3.add_kv_('a', 'b')
#     graph3.add_kv_('a', 0)
#     graph3.add_kv_('a', 1)
#     graph3.add_split_()
#     graph3.add_kv_('c', 2)
#     graph3.add_kv_('d', 'haha')
#     graph_info3 = ag.GraphInfo.from_data(indices=graph3.indices, values=graph3.formatted_values)
#
#     graph_info4 = ag.GraphInfo.batch([graph_info1, graph_info2])
#     assert ag.equal(graph_info3, graph_info4)
#
#     print(graph_info4.dictionary())
#     print(graph_info4.num_ecdfs)
#
#
# if __name__ == '__main__':
#     test_from_data()
#     test_reduce()
