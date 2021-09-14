import os
import pandas as pd
from all2graph import MetaInfo, progress_wrapper, Timer
from all2graph.json import JsonParser


def speed():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    csv_path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    nrows = 1000

    parser = JsonParser(
        'json', flatten_dict=True, local_index_names={'name'}, segment_value=True
    )

    with Timer('一遍过') as timer:
        df = pd.read_csv(csv_path, nrows=1000)
        graph, global_index_mapper, local_index_mappers = parser.parse(
            df, progress_bar=True
        )
        index_ids = list(global_index_mapper.values())
        for mapper in local_index_mappers:
            index_ids += list(mapper.values())
        meta_graph = MetaInfo.from_data(graph, index_nodes=index_ids, num_bins=None, progress_bar=True)
        used_time1 = timer.diff()
    print(meta_graph.meta_name.keys())

    with Timer('分片读取') as timer:
        meta_graphs = []
        chunks = list(pd.read_csv(csv_path, chunksize=nrows//10, nrows=nrows))
        for chunk in progress_wrapper(chunks):
            graph, global_index_mapper, local_index_mappers = parser.parse(chunk)
            index_ids = list(global_index_mapper.values())
            for mapper in local_index_mappers:
                index_ids += list(mapper.values())
            meta_graphs.append(MetaInfo.from_data(graph, index_nodes=index_ids, num_bins=None))
        used_time2 = timer.diff()

    with Timer('reduce') as timer:
        meta_graph2 = MetaInfo.reduce(meta_graphs, num_bins=None, progress_bar=True)
        used_time3 = timer.diff()
    print(meta_graph2.meta_name.keys())
    assert used_time3 < used_time1 and used_time3 < used_time2
    assert meta_graph == meta_graph2


if __name__ == '__main__':
    speed()
