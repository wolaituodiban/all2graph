import os
import shutil

import numpy as np
import pandas as pd
from all2graph import MetaInfo, JsonParser, Timer, JiebaTokenizer
from all2graph.factory import Factory


def test():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    csv_path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    nrows = 1000

    json_parser = JsonParser(
        'json', flatten_dict=True, local_index_names={'name'}, segment_value=True, tokenizer=JiebaTokenizer()
    )

    with Timer('工厂封装模式') as timer:
        factory = Factory(data_parser=json_parser)
        processes = os.cpu_count()

        meta_graph2 = factory.produce_meta_graph(
            csv_path, chunksize=int(np.ceil(nrows/processes)), progress_bar=True, processes=processes, nrows=1000
        )
        used_time1 = timer.diff()

    with Timer('原生模式') as timer:
        df = pd.read_csv(csv_path, nrows=1000)
        graph, global_index_mapper, local_index_mappers = json_parser.parse(
            df, progress_bar=True
        )
        index_ids = list(global_index_mapper.values())
        for mapper in local_index_mappers:
            index_ids += list(mapper.values())
        meta_graph1 = MetaInfo.from_data(graph, index_nodes=index_ids, progress_bar=True)
        used_time2 = timer.diff()

    assert meta_graph1 == meta_graph2
    assert used_time1 < used_time2

    # 测试保存文件
    save_path = os.path.join(path, 'test_data', 'graphs')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    factory.save_graphs(csv_path, save_path, chunksize=64, progress_bar=True, processes=None)
    shutil.rmtree(save_path)


if __name__ == '__main__':
    test()
