import os
import shutil

import all2graph as ag
import numpy as np
import pandas as pd
import torch


import platform
if 'darwin' in platform.system().lower():
    os.environ['OMP_NUM_THREADS'] = '1'


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


def test_csvdataset_v2():
    if os.path.exists('temp'):
        shutil.rmtree('temp')
    data = np.arange(9999)
    df = pd.DataFrame({'uid': data, 'data': data})

    meta_df = ag.split_csv(df, 'temp', chunksize=1000, meta_cols=['uid'], concat_chip=False)
    dataset = ag.data_parser.CSVDatasetV2(
        meta_df, data_parser=ParserMocker1(), raw_graph_parser=ParserMocker2(['data']), index_col=0)
    data_loader = dataset.build_dataloader(
        num_workers=3, prefetch_factor=1, shuffle=True, batch_size=16)
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


def test_dfdataset():
    data = np.arange(9999)
    df = pd.DataFrame({'uid': data, 'data': data})

    dataset = ag.data_parser.DFDataset(
        df=df, data_parser=ParserMocker1(), raw_graph_parser=ParserMocker2(['data']))
    data_loader = dataset.build_dataloader(
        num_workers=3, prefetch_factor=1, shuffle=True, batch_size=16)

    x, y = [], []
    for batch in ag.tqdm(data_loader):
        x.append(batch[0])
        y.append(batch[1])
    x = torch.cat(x)
    y = torch.cat(y)


if __name__ == '__main__':
    test_csvdataset_v2()
    test_dfdataset()
