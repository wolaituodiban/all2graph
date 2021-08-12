import os
import pandas as pd
from all2graph.meta_graph import JsonMetaGraph
from all2graph.meta_node import JsonValue


def test_json_graph():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    df = pd.read_csv(path)
    df = df.iloc[:3000]
    meta_graph = JsonMetaGraph.from_data(
        df.dateupdated,
        df.json,
        index_names={
            'descriptions',
            'imageurls',
            'reviews',
            'sourceURLs',
            'sourceurls'
            'text'
        }
    )
    for k, v in meta_graph.nodes.items():
        if isinstance(v, JsonValue):
            if 'string' in v.value_dist:
                print(k, len(v.value_dist['string']), v.value_dist['string'].max_len)
        else:
            print(k, v)


if __name__ == '__main__':
    test_json_graph()
    print('测试JsonGraph成功')
