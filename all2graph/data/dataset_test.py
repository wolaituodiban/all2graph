import os
import platform
import shutil
import json

import pandas as pd
import all2graph as ag

if 'darwin' in platform.system().lower():
    os.environ['OMP_NUM_THREADS'] = '1'


def foo(x):
    return x


def test_csv_dataset():
    if os.path.exists('temp'):
        shutil.rmtree('temp')

    path_df = ag.split_csv(df, 'temp', chunksize=100, drop_cols=['json'])
    dataset = ag.data.CSVDataset(path_df, parser=parser_wrapper, func=foo)
    data_loader = dataset.dataloader(num_workers=2, shuffle=True, batch_size=16)

    num_samples = 0
    x, y = None, None
    for batch in ag.tqdm(data_loader):
        x, y = batch
        num_samples += x.num_samples
    print(x, y)
    assert num_samples == 1000
    shutil.rmtree('temp')
    os.remove('temp_path.zip')


def test_df_dataset():
    dataset = ag.data.DFDataset(df, parser=parser_wrapper)
    data_loader = dataset.dataloader(num_workers=2, shuffle=True, batch_size=16)

    num_samples = 0
    x, y = None, None
    for batch in ag.tqdm(data_loader):
        x, y = batch
        num_samples += x.num_samples
    print(x, y)
    assert num_samples == 1000


def test_graph_dataset():
    if os.path.exists('temp'):
        shutil.rmtree('temp')
    path_df = parser_wrapper.save(df, 'temp')
    dataset = ag.data.GraphDataset(path_df)
    data_loader = dataset.dataloader(num_workers=2, shuffle=True, batch_size=16)
    num_samples = 0
    x, y = None, None
    for batch in ag.tqdm(data_loader):
        x, y = batch
        num_samples += x.num_samples
    print(x, y)
    assert num_samples == 1000
    shutil.rmtree('temp')
    os.remove('temp_path.zip')


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
        json_col='json', time_col='crt_dte', time_format='%Y-%m-%d', targets=['m3_ovd_30'], local_foreign_key_types={'ord_no'})
    raw_graph = json_parser(df.iloc[:1], disable=False)
    meta_info = ag.MetaInfo.from_data(raw_graph)
    graph_parser = ag.GraphParser.from_data(meta_info)
    parser_wrapper = ag.ParserWrapper(data_parser=json_parser, graph_parser=graph_parser)

    test_csv_dataset()
    test_df_dataset()
    # test_graph_dataset()
