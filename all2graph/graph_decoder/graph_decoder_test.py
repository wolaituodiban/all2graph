import os
import json
import numpy as np
import pandas as pd
from all2graph.resolvers import JsonResolver
from all2graph.graph_decoder import GraphDecoder


def test():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    df = pd.read_csv(path)

    resolver = JsonResolver(
        dict_pred_degree=0, list_pred_degree=0, list_inner_degree=0, r_list_inner_degree=0, global_index_names={'name'}
    )
    graph, _ = resolver.resolve('graph', list(map(json.loads, df.json)))
    decoder = GraphDecoder.from_data(graph, num_bins=None)
    print(len(decoder.meta_string))

    print('这里')
    decoders = []
    weights = []
    for chunk in pd.read_csv(path, chunksize=1000):
        graph, _ = resolver.resolve('graph', list(map(json.loads, chunk.json)))
        decoders.append(GraphDecoder.from_data(graph, num_bins=None))
        weights.append(graph.num_nodes)

    print('开始reduce')
    decoder2 = GraphDecoder.reduce(decoders, num_bins=None)
    assert decoder.meta_numbers == decoder2.meta_numbers


if __name__ == '__main__':
    test()
