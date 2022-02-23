import os
import all2graph as ag
import pandas as pd


def test_save_and_load():
    sample = {'a': 'b'}
    df = pd.DataFrame({'id': [0], 'json': [sample], 'day': [None]})
    json_parser = ag.json.JsonParser('json', time_col='day', error=False, warning=False)
    raw_graph, *_ = json_parser.parse(df)
    meta_info = ag.MetaInfo.from_data(raw_graph)
    raw_graph_parser = ag.RawGraphParser.from_data(meta_info, targets=['target'])

    path = 'temp.dgl.graphs'
    graph = raw_graph_parser.parse(raw_graph)
    print(graph.edge_key, raw_graph_parser.etype_mapper)
    graph.save(path)
    graph2, labels = ag.graph.Graph.load(path)
    os.remove(path)
    assert graph.__eq__(graph2, debug=True)


if __name__ == '__main__':
    test_save_and_load()
