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
        dict_pred_degree=0, list_pred_degree=0, list_inner_degree=0, r_list_inner_degree=0, global_index_names={'name'}
    )
    graph, _ = resolver.resolve('graph', list(map(json.loads, df.json)), progress_bar=True)
    decoder = GraphDecoder.from_data(graph, num_bins=None, progress_bar=True)
    used_time1 = time.time() - start_time1
    print(len(decoder.meta_string))
    print(decoder.meta_name.keys())

    decoders = []
    chunks = list(pd.read_csv(path, chunksize=1000))
    start_time2 = time.time()
    for chunk in Progress(chunks):
        graph, _ = resolver.resolve('graph', list(map(json.loads, chunk.json)))
        decoders.append(GraphDecoder.from_data(graph, num_bins=None))
    used_time2 = time.time() - start_time2

    print('开始reduce')
    start_time3 = time.time()
    decoder2 = GraphDecoder.reduce(decoders, num_bins=None, progress_bar=True)
    used_time3 = time.time() - start_time3
    print(used_time1, used_time2, used_time3)
    print(decoder2.meta_name.keys())
    assert decoder.meta_numbers == decoder2.meta_numbers
    assert used_time3 < used_time1 < used_time2


if __name__ == '__main__':
    test()
