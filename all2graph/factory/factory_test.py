import os
import json
import shutil
import time
import pickle

import numpy as np
import pandas as pd
from all2graph import MetaGraph, JsonPathTree, JsonParser
from all2graph.factory import Factory


def test():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    csv_path = os.path.join(path, 'test_data', 'MensShoePrices.csv')

    preprocessor = JsonPathTree('json')
    json_parser = JsonParser(
        root_name=preprocessor.json_col, flatten_dict=True, local_index_names={'name'}, segmentation=True
    )

    # 工厂封装模式
    start_time1 = time.time()
    factory = Factory(
        data_parser=json_parser, preprocessor=preprocessor, graph_transer_config=dict(
            min_df=0.01, max_df=0.95, top_k=100, top_method='max_tfidf', segment_name=False,
        )
    )
    processes = os.cpu_count()

    meta_graph2 = factory.produce_meta_graph(
        csv_path, chunksize=int(np.ceil(10000/processes)), progress_bar=True, processes=processes,
    )
    used_time1 = time.time() - start_time1
    print(used_time1)

    # 原生模式
    start_time2 = time.time()
    df = pd.read_csv(csv_path)
    graph, global_index_mapper, local_index_mappers = json_parser.parse(
        preprocessor(df), progress_bar=True
    )
    index_ids = list(global_index_mapper.values())
    for mapper in local_index_mappers:
        index_ids += list(mapper.values())
    meta_graph1 = MetaGraph.from_data(graph, index_nodes=index_ids, progress_bar=True)
    used_time2 = time.time() - start_time2
    print(used_time1, used_time2)
    assert meta_graph1 == meta_graph2
    assert used_time1 < used_time2

    # 测试保存文件
    save_path = os.path.join(path, 'test_data', 'graphs')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    factory.save_graphs(csv_path, save_path, chunksize=64, progress_bar=True, processes=None)


if __name__ == '__main__':
    test()
