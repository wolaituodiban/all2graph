import os
import pandas as pd
from all2graph.meta_graph import JsonGraph
from all2graph.meta_graph.meta_node import JsonValue


def test_json_graph():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    meta_graphs = [JsonGraph.from_data(
            chunk.dateupdated,
            chunk.json,
            index_names={
                'descriptions',
                'imageurls',
                'reviews',
                'sourceURLs',
                'sourceurls'
                'text'
            }
        ) for chunk in pd.read_csv(path, chunksize=100, nrows=200)
    ]
    print('reduce开始')
    meta_graph = JsonGraph.reduce(meta_graphs)

    for k, v in meta_graph.nodes.items():
        if isinstance(v, JsonValue):
            if 'string' in v.value_dist:
                print(k, len(v.value_dist['string']), v.value_dist['string'].max_len)
        else:
            print(k, v)


if __name__ == '__main__':
    test_json_graph()
    print('测试JsonGraph成功')
