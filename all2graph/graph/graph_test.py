from all2graph import Graph


def test():
    graph = Graph(component_ids=[0, 0, 1, 1, 1, 2], names=['a', 'b', 'a', 'c', 'b', 'a'],
                  values=[1, None, 'a', 'a', 'c', 'b'], preds=[0, 1, 0, 2, 1, 2, 2, 1], succs=[1, 2, 0, 1, 1, 1, 1, 1])
    node_df = graph.node_df()
    edge_df = graph.edge_df(node_df)
    graph2 = Graph.from_data(node_df=node_df, edge_df=edge_df)
    assert graph == graph2


if __name__ == '__main__':
    test()
