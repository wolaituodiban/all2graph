import os
import json
import time
import pandas as pd
from toad.utils.progress import Progress
from all2graph.resolver import JsonResolver
from all2graph.decoder import GraphDecoder


def test():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    df = pd.read_csv(path)

    start_time1 = time.time()
    resolver = JsonResolver(
        dict_pred_degree=0, list_pred_degree=0, list_inner_degree=0, r_list_inner_degree=0, local_index_names={'name'}
    )
    graph, global_index_mapper, local_index_mappers = resolver.resolve(
        'graph', list(map(json.loads, df.json)), progress_bar=True
    )
    index_ids = list(global_index_mapper.values())
    for mapper in local_index_mappers:
        index_ids += list(mapper.values())
    decoder = GraphDecoder.from_data(graph, drop_nodes=index_ids, num_bins=None, progress_bar=True)
    used_time1 = time.time() - start_time1
    print(decoder.meta_name.keys())

    decoders = []
    chunks = list(pd.read_csv(path, chunksize=1000))
    start_time2 = time.time()
    for chunk in Progress(chunks):
        graph, global_index_mapper, local_index_mappers = resolver.resolve('graph', list(map(json.loads, chunk.json)))
        index_ids = list(global_index_mapper.values())
        for mapper in local_index_mappers:
            index_ids += list(mapper.values())
        decoders.append(GraphDecoder.from_data(graph, drop_nodes=index_ids, num_bins=None))
    used_time2 = time.time() - start_time2

    print('开始reduce')
    start_time3 = time.time()
    decoder2 = GraphDecoder.reduce(decoders, num_bins=None, progress_bar=True)
    used_time3 = time.time() - start_time3
    print(used_time1, used_time2, used_time3)
    print(decoder2.meta_name.keys())

    assert used_time3 < used_time1 and used_time3 < used_time2
    assert decoder == decoder2


if __name__ == '__main__':
    test()
