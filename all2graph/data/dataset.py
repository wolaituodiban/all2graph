import sys
import traceback
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from pandas.errors import ParserError
from torch.utils.data import Dataset as _Dataset, DataLoader

from .sampler import PartitionSampler
from ..graph import Graph
from ..parsers import DataParser, RawGraphParser
from ..utils import tqdm, iter_files


class Dataset(_Dataset):
    def __init__(self, data_parser: DataParser, raw_graph_parser: RawGraphParser, **kwargs):
        self.data_parser = data_parser
        self.raw_graph_parser = raw_graph_parser
        self.kwargs = kwargs

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item) -> pd.DataFrame:
        raise NotImplementedError

    def read_csv(self, path):
        return pd.read_csv(path, **self.kwargs)

    def collate_fn(self, batches: List[pd.DataFrame]) -> Tuple[Graph, Dict[str, torch.Tensor]]:
        df = pd.concat(batches)
        graph = self.data_parser.parse(df, disable=True)[0]
        graph = self.raw_graph_parser.parse(graph)
        label = self.data_parser.gen_targets(df, self.raw_graph_parser.targets)
        return graph, label

    def enable_preprocessing(self):
        self.data_parser.enable_preprocessing()

    def disable_preprocessing(self):
        self.data_parser.disable_preprocessing()

    def set_filter_key(self, x):
        self.raw_graph_parser.set_filter_key(x)


class CSVDataset(Dataset):
    def __init__(
            self, src, data_parser: DataParser, raw_graph_parser: RawGraphParser, chunksize=64,
            shuffle=False, disable=True, error=True, warning=True, **kwargs):
        print('CSVDataset is depreciated, please use CSVDatasetV2')
        super().__init__(data_parser=data_parser, raw_graph_parser=raw_graph_parser, **kwargs)

        paths: List[Tuple[str, int]] = []
        for path in tqdm(
                iter_files(src, error=error, warning=warning), disable=disable, postfix='checking files'):
            try:
                df = self.read_csv(path)
                row_nums = list(range(df.shape[0]))
                if shuffle:
                    np.random.shuffle(row_nums)
                for row_num in row_nums:
                    paths.append((path, row_num))
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except ParserError:
                if error:
                    raise ValueError('read "{}" encountered error'.format(path))
                elif warning:
                    print('read "{}" encountered error'.format(path), file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)

        self.paths: List[Dict[str, List[int]]] = []
        for i in range(0, len(paths), chunksize):
            temp = {}
            for path, row_num in paths[i:i + chunksize]:
                if path in temp:
                    temp[path].append(row_num)
                else:
                    temp[path] = [row_num]
            self.paths.append(temp)
        # 转化成pandas类型，似乎可以减小datalodaer多进程的开销
        self.paths = pd.Series(self.paths)

    @property
    def chunksize(self):
        return sum(map(len, self.paths[0].values()))

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, item) -> pd.DataFrame:
        dfs = []
        for path, row_nums in self.paths.iloc[item].items():
            df = self.read_csv(path)
            df = df.iloc[row_nums]
            dfs.append(df)
        df = pd.concat(dfs)
        return df


class GraphDataset(_Dataset):
    def __init__(self, src, error=True, warning=True):
        self.path = pd.Series(iter_files(src, error=error, warning=warning))

    def __len__(self):
        return self.path.shape[0]

    def __getitem__(self, item) -> Graph:
        return Graph.load(self.path.iloc[item])


class CSVDatasetV2(Dataset):
    def __init__(
            self, src: pd.DataFrame, data_parser: DataParser, raw_graph_parser: RawGraphParser, **kwargs):
        """

        Args:
            src: 长度为样本数量，需要有一列path
                例如  path
                    1.csv
                    1.csv
                    2.csv
                    2.csv
                    2.csv
            data_parser:
            raw_graph_parser:
            **kwargs:
        """
        super().__init__(data_parser=data_parser, raw_graph_parser=raw_graph_parser, **kwargs)
        path = src.groupby('path').agg({'path': 'count'})
        path.columns = ['lines']
        path['ub'] = path['lines'].cumsum()
        path['lb'] = path['ub'].shift(fill_value=0)
        self._path = path
        self.__partitions = {}

    def __len__(self):
        return self._path['ub'].iloc[-1]

    # def _get_partition_num(self, item):
    #     for i, ub in enumerate(self._path['ub']):
    #         if item < ub:
    #             return i
    #     raise IndexError('out of bound')

    def _get_partition_num(self, item, left=0, right=None):
        assert 0 <= item < len(self), 'out of bound'
        right = right or self._path.shape[0]
        mid = (left + right) // 2
        if self._path.iloc[mid]['lb'] <= item:
            if item < self._path.iloc[mid]['ub']:
                return mid
            else:
                return self._get_partition_num(item, left=mid, right=right)
        else:
            return self._get_partition_num(item, left=left, right=mid)

    def _get_partition(self, partition_num):
        if partition_num not in self.__partitions:
            # print(torch.utils.data.get_worker_info().id, partition_num)
            self.__partitions = {
                partition_num: pd.read_csv(self._path.index[partition_num], **self.kwargs)
            }
        return self.__partitions[partition_num]

    def __getitem__(self, item) -> pd.DataFrame:
        partition_num = self._get_partition_num(item)
        # print(torch.utils.data.get_worker_info().id, partition_num)
        partition = self._get_partition(partition_num)
        df = partition.iloc[[item - self._path['lb'].iloc[partition_num]]]
        return df

    def batch_sampler(self, num_workers: int, shuffle=False, batch_size=1):
        indices = []
        for i, row in self._path.iterrows():
            indices.append(list(range(row['lb'], row['ub'])))
        return PartitionSampler(indices=indices, num_workers=num_workers, shuffle=shuffle, batch_size=batch_size)

    def build_dataloader(self, num_workers: int, shuffle=False, batch_size=1, **kwargs) -> DataLoader:
        sampler = self.batch_sampler(shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        return DataLoader(
            self, collate_fn=self.collate_fn, batch_sampler=sampler, num_workers=num_workers, **kwargs)
