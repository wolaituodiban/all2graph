import os
import shutil
import numpy as np
import pandas as pd
import all2graph as ag
from all2graph import MetaInfo, JsonParser, Timer, JiebaTokenizer, progress_wrapper
from all2graph.factory import Factory


def test_analyse():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    csv_path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    nrows = 1000

    json_parser = JsonParser(
        'json', flatten_dict=True, local_id_keys={'name'}, segment_value=True, tokenizer=JiebaTokenizer()
    )

    with Timer('工厂封装模式') as timer:
        factory = Factory(data_parser=json_parser)
        processes = os.cpu_count()

        meta_graph2 = factory.analyse(
            csv_path, chunksize=int(np.ceil(nrows/processes)), progress_bar=True, processes=processes, nrows=1000
        )
        used_time1 = timer.diff()
    print(factory)

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


def test_produce_dataloader():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    csv_path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    save_path = os.path.join(path, 'test_data', 'temp')
    nrows = 1000

    cpu_count = os.cpu_count()
    json_parser = JsonParser(
        'json', flatten_dict=True, local_id_keys={'name'}, segment_value=True, tokenizer=JiebaTokenizer()
    )
    factory = Factory(data_parser=json_parser)
    factory.analyse(
        csv_path, chunksize=int(nrows//cpu_count), progress_bar=True, processes=cpu_count, nrows=nrows
    )
    with ag.Timer('v1'):
        dataloader = factory.produce_dataloader(
            csv_path, dst=save_path, csv_configs={'nrows': nrows}, num_workers=cpu_count)
        for _ in progress_wrapper(dataloader):
            pass
    with ag.Timer('v2'):
        dataloader = factory.produce_dataloader(
            save_path, csv_configs={'nrows': nrows}, num_workers=cpu_count, version=2, temp_file=True)
        for _ in progress_wrapper(dataloader):
            pass
    shutil.rmtree(save_path)


def test_produce_model():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    csv_path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    nrows = 1000

    cpu_count = os.cpu_count()
    json_parser = JsonParser(
        'json', flatten_dict=True, local_id_keys={'name'}, segment_value=True, tokenizer=JiebaTokenizer()
    )
    factory = Factory(data_parser=json_parser)
    factory.analyse(
        csv_path, chunksize=int(nrows//cpu_count), progress_bar=True, processes=cpu_count, nrows=nrows
    )
    model = factory.produce_model(d_model=8, nhead=2, num_layers=[2], mock=True)
    print(model)


if __name__ == '__main__':
    test_analyse()
    test_produce_dataloader()
    test_produce_model()
