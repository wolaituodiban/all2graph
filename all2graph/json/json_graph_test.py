import os
import json
import pandas as pd
from all2graph.json import JsonGraph
from tqdm import tqdm


def test_json_graph(df):
    json_graph = JsonGraph()
    for value in tqdm(df.json):
        json_graph.insert_json('json', json.loads(value))
    print(json_graph.num_nodes, json_graph.num_edges)
    print(json_graph.names[:100])
    print(json_graph.edges[:100])


if __name__ == '__main__':
    path = os.path.join('test_data', 'MensShoePrices.csv')
    df = pd.read_csv(path)
    test_json_graph(pd.concat([df] * 10))
