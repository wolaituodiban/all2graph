import os
import shutil

import all2graph as ag
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


import platform
if 'darwin' in platform.system().lower():
    os.environ['OMP_NUM_THREADS'] = '1'


def test_dataset():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    csv_path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    df = pd.read_csv(csv_path)

    json_parser = ag.json.JsonParser(
        'json', flatten_dict=True, local_id_keys={'name'}, segment_value=True
    )
    raw_graph_parser = ag.RawGraphParser.from_data(
        ag.MetaInfo.from_data(json_parser.parse(df, disable=False)[0], disable=False)
    )
    # 测试保存文件
    save_path = os.path.join(path, 'test_data', 'temp')
    ag.split_csv(df, save_path, chunksize=5, disable=False, zip=True, concat_chip=True)
    graph_paths = [os.path.join(save_path, file) for file in os.listdir(save_path)]

    # 开始测试
    with ag.Timer('dataset'):
        dataset = ag.data.CSVDataset(
            graph_paths, data_parser=json_parser, raw_graph_parser=raw_graph_parser,
            chunksize=32, shuffle=True, disable=False)
        num_rows2 = []
        for _ in ag.tqdm(dataset):
            num_rows2.append(_.shape[0])
    shutil.rmtree(save_path)
    temp = []
    for a in dataset.paths:
        for b, c in a.items():
            for d in c:
                temp.append((b, d))
    # 测试每一个样本不重复不遗漏
    assert len(set(temp)) == df.shape[0]
    assert df.shape[0] == sum(num_rows2)
    # 测试batchsize正确
    assert set(num_rows2[:-1]) == {dataset.chunksize}, (num_rows2, dataset.chunksize)


def test_dataset_v2():
    class ParserMocker1:
        def parse(self, df, **kwargs):
            return df.values,

        def gen_targets(self, df, targets):
            return torch.tensor(df[targets].values)

    class ParserMocker2:
        def __init__(self, targets):
            self.targets = targets

        def parse(self, x):
            return torch.tensor(x)

    if os.path.exists('temp'):
        shutil.rmtree('temp')
    data = np.arange(9999)
    df = pd.DataFrame({'uid': data, 'data': data})

    meta_df = ag.split_csv(df, 'temp', chunksize=1000, meta_cols=['uid'], concat_chip=False)
    dataset = ag.data.CSVDatasetV2(
        meta_df, data_parser=ParserMocker1(), raw_graph_parser=ParserMocker2(['data']), index_col=0)
    data_loader = DataLoader(
        dataset, num_workers=3, collate_fn=dataset.collate_fn, prefetch_factor=1,
        batch_sampler=dataset.build_sampler(num_workers=3, shuffle=True, batch_size=16))
    #
    # for i in data_loader.sampler:
    #     print(i, dataset._get_partition_num(i))

    x, y = [], []
    for batch in ag.tqdm(data_loader):
        x.append(batch[0])
        y.append(batch[1])
    x = torch.cat(x)
    y = torch.cat(y)

    shutil.rmtree('temp')


if __name__ == '__main__':
    test_dataset()
    test_dataset_v2()
