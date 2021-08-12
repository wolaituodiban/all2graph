import os
import json
import pandas as pd
from all2graph.graph import Graph
from toad.utils.progress import Progress


def test_json_graph():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    df = pd.read_csv(path)
    json_graph = Graph()
    for i, value in enumerate(Progress(df.json.values)):
        json_graph.insert_patch(i, 'graph', json.loads(value))
    print(json_graph.num_nodes, json_graph.num_edges)
    print(json_graph.patch_ids[:100])
    print(json_graph.patch_ids[-100:])
    print(json_graph.names[:100])
    print(json_graph.preds[:100])
    print(json_graph.succs[:100])


if __name__ == '__main__':
    test_json_graph()
