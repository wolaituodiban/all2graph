import os
import json
import time
import pandas as pd
from toad.utils.progress import Progress
from all2graph.resolver import JsonResolver
from all2graph.meta_graph import MetaGraph


def test():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    csv_path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    df = pd.read_csv(csv_path)

    start_time1 = time.time()
    resolver = JsonResolver(
        flatten_dict=True, local_index_names={'name'}, segmentation=True
    )
    graph, global_index_mapper, local_index_mappers = resolver.resolve(
        'graph', list(map(json.loads, df.json)), progress_bar=True
    )
    index_ids = list(global_index_mapper.values())
    for mapper in local_index_mappers:
        index_ids += list(mapper.values())
    meta_graph = MetaGraph.from_data(graph, drop_nodes=index_ids, num_bins=None, progress_bar=True)
    used_time1 = time.time() - start_time1
    print(meta_graph.meta_name.keys())
    with open(os.path.join(path, 'test_data', 'meta_graph.json'), 'w') as file:
        json.dump(meta_graph.to_json(), file)

    print('开始分片读取')
    meta_graphs = []
    chunks = list(pd.read_csv(csv_path, chunksize=1000))
    start_time2 = time.time()
    for chunk in Progress(chunks):
        graph, global_index_mapper, local_index_mappers = resolver.resolve('graph', list(map(json.loads, chunk.json)))
        index_ids = list(global_index_mapper.values())
        for mapper in local_index_mappers:
            index_ids += list(mapper.values())
        meta_graphs.append(MetaGraph.from_data(graph, drop_nodes=index_ids, num_bins=None))
    used_time2 = time.time() - start_time2

    print('开始reduce')
    start_time3 = time.time()
    meta_graph2 = MetaGraph.reduce(meta_graphs, num_bins=None, progress_bar=True)
    used_time3 = time.time() - start_time3
    print(used_time1, used_time2, used_time3)
    print(meta_graph2.meta_name.keys())
    with open(os.path.join(path, 'test_data', 'meta_graph2.json'), 'w') as file:
        json.dump(meta_graph2.to_json(), file)
    assert used_time3 < used_time1 and used_time3 < used_time2
    assert meta_graph == meta_graph2


if __name__ == '__main__':
    test()
