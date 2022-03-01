import os
import shutil
import numpy as np
import pandas as pd
import all2graph as ag


def test_analyse():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    csv_path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    nrows = 1000

    json_parser = ag.json.JsonParser(
        'json', time_col='day', local_id_keys={'name'}, tokenizer=ag.JiebaTokenizer()
    )

    with ag.Timer('工厂封装模式') as timer:
        factory = ag.Factory(data_parser=json_parser)
        processes = os.cpu_count()

        meta_graph2 = factory.analyse(
            csv_path, chunksize=int(np.ceil(nrows/processes)), disable=True, processes=processes, nrows=1000
        )
        # used_time1 = timer.diff()
    print(factory)

    with ag.Timer('原生模式') as timer:
        df = pd.read_csv(csv_path, nrows=1000)
        graph, global_index_mapper, local_index_mappers = json_parser.__call__(df, disable=False)
        index_ids = list(global_index_mapper.values())
        for mapper in local_index_mappers:
            index_ids += list(mapper.values())
        meta_graph1 = ag.MetaInfo.from_data(graph, index_nodes=index_ids, disable=False)
        # used_time2 = timer.diff()

    assert meta_graph1 == meta_graph2
    # assert used_time1 < used_time2


def test_scale():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    csv_path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    df = pd.read_csv(csv_path)

    json_parser = ag.json.JsonParser('json', time_col='day', local_id_keys={'name'}, tokenizer=ag.JiebaTokenizer())

    factory = ag.Factory(
        data_parser=json_parser,
        raw_graph_parser_config=dict(scale_method='minmax')
    )
    processes = os.cpu_count()

    factory.analyse(
        df, chunksize=int(np.ceil(df.shape[0] / processes)), disable=True, processes=processes, nrows=1000
    )
    print(factory.produce_graph_and_label(df.iloc[:10]))


def test_produce_dataloader():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    csv_path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    save_path = os.path.join(path, 'test_data', 'temp')
    nrows = 1000

    cpu_count = os.cpu_count()
    try:
        json_parser = ag.json.JsonParser('json', time_col='day',  local_id_keys={'name'}, tokenizer=ag.JiebaTokenizer())
        factory = ag.Factory(data_parser=json_parser, raw_graph_parser_config=dict(filter_key=True))
        factory.analyse(
            csv_path, chunksize=int(nrows//cpu_count), disable=True, processes=cpu_count, nrows=nrows
        )
        configs = {'nrows': nrows}
        meta_df = ag.split_csv(csv_path, save_path, chunksize=100, **configs)

        with ag.Timer('csv'):
            dataloader = factory.produce_dataloader(
                meta_df=meta_df, csv_configs=configs, num_workers=1)
            assert isinstance(dataloader.dataset, ag.data_parser.CSVDatasetV2), dataloader.dataset
            for _ in ag.tqdm(dataloader):
                pass
            shutil.rmtree(save_path)

        with ag.Timer('df'):
            dataloader = factory.produce_dataloader(
                df=pd.read_csv(csv_path, **configs), num_workers=1)
            assert isinstance(dataloader.dataset, ag.data_parser.DFDataset), dataloader.dataset
            for _ in ag.tqdm(dataloader):
                pass

        meta_df = factory.save(csv_path, save_path, processes=cpu_count, chunksize=10)
        with ag.Timer('graph'):
            dataloader = factory.produce_dataloader(meta_df=meta_df, num_workers=1, graph=True)
            assert isinstance(dataloader.dataset, ag.data_parser.GraphDataset), dataloader.dataset
            for _ in ag.tqdm(dataloader):
                pass
    finally:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)


def test_produce_model():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    csv_path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    nrows = 1000

    cpu_count = os.cpu_count()
    json_parser = ag.json.JsonParser('json', time_col='day', local_id_keys={'name'}, tokenizer=ag.JiebaTokenizer())
    factory = ag.Factory(data_parser=json_parser)
    factory.analyse(
        csv_path, chunksize=int(nrows//cpu_count), disable=True, processes=cpu_count, nrows=nrows
    )
    model = factory.produce_model(d_model=8, nhead=2, num_layers=[2], mock=True)
    print(model)


if __name__ == '__main__':
    test_analyse()
    test_scale()
    test_produce_dataloader()
    test_produce_model()
