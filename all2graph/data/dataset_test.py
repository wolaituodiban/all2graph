import os
import platform
import shutil
import json

import pandas as pd
import all2graph as ag

if 'darwin' in platform.system().lower():
    os.environ['OMP_NUM_THREADS'] = '1'


def test_csv_dataset():
    if os.path.exists('temp'):
        shutil.rmtree('temp')

    path_df = ag.split_csv(df, 'temp', chunksize=100, drop_cols=['json'])
    dataset = ag.data.CSVDataset(path_df, parser=parser_wrapper)
    data_loader = dataset.dataloader(num_workers=2, shuffle=True, batch_size=16)

    num_samples = 0
    x, y = None, None
    for batch in ag.tqdm(data_loader):
        x, y = batch
        num_samples += x.graph.num_nodes('m3_ovd_30')
    print(x, y)
    assert num_samples == 1000
    shutil.rmtree('temp')
    os.remove('temp_path.csv')


def test_df_dataset():
    dataset = ag.data.DFDataset(df, parser=parser_wrapper)
    data_loader = dataset.dataloader(num_workers=2, shuffle=True, batch_size=16)

    num_samples = 0
    x, y = None, None
    for batch in ag.tqdm(data_loader):
        x, y = batch
        num_samples += x.graph.num_nodes('m3_ovd_30')
    print(x, y)
    assert num_samples == 1000


def test_graph_dataset():
    if os.path.exists('temp'):
        shutil.rmtree('temp')
    path_df = parser_wrapper.save(df, 'temp')
    dataset = ag.data.GraphDataset(path_df, parser_wrapper)
    data_loader = dataset.dataloader(num_workers=2, shuffle=True, batch_size=16)
    num_samples = 0
    x, y = None, None
    for batch in ag.tqdm(data_loader):
        x, y = batch
        num_samples += x.graph.num_nodes('m3_ovd_30')
    print(x, y)
    assert num_samples == 1000
    shutil.rmtree('temp')
    os.remove('temp_path.csv')


if __name__ == '__main__':
    data = [
        {
            'ord_no': 'CH202007281033864',
            'bsy_typ': 'CASH',
        },
        {
            'ord_no': 'CH202007281033864',
            'stg_no': '1',
        },
    ] * 100
    df = pd.DataFrame({'json': [json.dumps(data)], 'crt_dte': '2020-10-09'})
    df = pd.concat([df] * 1000)
    json_parser = ag.JsonParser(
        json_col='json', time_col='crt_dte', time_format='%Y-%m-%d', targets=['m3_ovd_30'], lid_keys={'ord_no'})
    raw_graph = json_parser(df, disable=False)
    print(raw_graph)
    print(raw_graph.num_samples)
    meta_info = raw_graph.meta_info()
    graph_parser = ag.GraphParser.from_data(meta_info)
    post_parser = ag.PostParser()
    parser_wrapper = ag.ParserWrapper(data_parser=json_parser, graph_parser=graph_parser, post_parser=post_parser)

    test_csv_dataset()
    test_df_dataset()
    test_graph_dataset()
