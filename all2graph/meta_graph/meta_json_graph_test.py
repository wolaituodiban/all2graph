import os
import pandas as pd
from all2graph.meta_graph import MetaJsonGraph
from all2graph.meta_graph.meta_node import MetaJsonValue


def test_json_graph():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
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
            chunksize=100,
            nrows=200
        )
    ]
    print('reduce开始')
    meta_graph = MetaJsonGraph.reduce(meta_graphs)

    for k, v in meta_graph.nodes.items():
        if isinstance(v, MetaJsonValue):
            print(v.to_discrete().prob)
            if 'string' in v.meta_data:
                print(k, len(v.meta_data['string']), v.meta_data['string'].max_str_len)
        else:
            print(k, v)

    # df = pd.read_csv(path)
    # meta_graph = MetaJsonGraph.from_data(
    #     df.dateupdated,
    #     df.json,
    #     index_names={
    #         'descriptions',
    #         'imageurls',
    #         'reviews',
    #         'sourceURLs',
    #         'sourceurls'
    #         'text'
    #     }
    # )
    # for k, v in meta_graph.nodes.items():
    #     if isinstance(v, MetaJsonValue):
    #         print(v.to_discrete().prob)
    #         if 'string' in v.value_dist:
    #             print(k, len(v.value_dist['string']), v.value_dist['string'].max_len)
    #     else:
    #         print(k, v)


if __name__ == '__main__':
    test_json_graph()
    print('测试JsonGraph成功')
