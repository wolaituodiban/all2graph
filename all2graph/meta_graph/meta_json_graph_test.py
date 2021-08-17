import os
import pandas as pd
import time
from all2graph.meta_graph import MetaJsonGraph, MetaJsonValue


def test_json_graph():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.join(path, 'test_data', 'MensShoePrices.csv')

    start_time1 = time.time()
    meta_graphs = [MetaJsonGraph.from_data(
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
        ) for chunk in pd.read_csv(
            path,
            chunksize=1000,
            # nrows=200
        )
    ]
    print('reduce开始')
    meta_graph1 = MetaJsonGraph.reduce(meta_graphs)
    use_time1 = time.time() - start_time1

    start_time2 = time.time()
    df = pd.read_csv(path)
    meta_graph2 = MetaJsonGraph.from_data(
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
    use_time2 = time.time() - start_time2
    print(use_time1, use_time2)
    # todo 找出错误
    for name in meta_graph1.nodes:
        for k in meta_graph1.nodes[name]:
            if meta_graph1.nodes[name][k] != meta_graph2.nodes[name][k]:
                print(name, k)


if __name__ == '__main__':
    test_json_graph()
    print('测试JsonGraph成功')
