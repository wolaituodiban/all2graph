from abc import abstractmethod
from typing import List, Tuple, Dict, Any

import pandas as pd
import torch
from torch.utils.data import Dataset as _Dataset, DataLoader

from .sampler import PartitionSampler


class Dataset(_Dataset):
    @abstractmethod
    def collate_fn(self, batches):
        raise NotImplementedError

    def dataloader(self, **kwargs) -> DataLoader:
        return DataLoader(self, collate_fn=self.collate_fn, **kwargs)


class ParserDataset(Dataset):
    def __init__(self, parser, func):
        """
        parser: instance, 需要实现两个方法: __call__(self, df: pd.DataFrame)和get_targets(df: pd.DataFrame))
        func: dataframe的预处理函数
        """
        self.parser = parser
        self.func = func

    def collate_fn(self, batches: List[pd.DataFrame]) -> Tuple[Any, Dict[str, torch.Tensor]]:
        df = pd.concat(batches)
        if self.func is not None:
            df = self.func(df)
        graph = self.parser(df)
        return graph, self.parser.get_targets(df)


class PartitionDataset(Dataset):
    def __init__(self, path: pd.DataFrame):
        """

        Args:
            path: 长度为样本数量, 需要有一列path
                例如  path
                    1.csv
                    1.csv
                    2.csv
                    2.csv
                    2.csv
        """
        path = path.groupby('path').agg({'path': 'count'})
        path.columns = ['lines']
        path['ub'] = path['lines'].cumsum()
        path['lb'] = path['ub'].shift(fill_value=0)
        self._path = path
        self._partitions = {}

    def __len__(self):
        return self._path['ub'].iloc[-1]

    @abstractmethod
    def read_file(self, path):
        raise NotImplementedError

    @abstractmethod
    def get_partition_len(self, partition) -> int:
        raise NotImplementedError

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
        if partition_num not in self._partitions:
            # print(torch.utils.data.get_worker_info().id, partition_num)
            partition = self.read_file(self._path.index[partition_num])
            self._partitions = {
                partition_num: [partition, self.get_partition_len(partition), 0]
            }

        partitions = self._partitions[partition_num][0]
        # partiton计数器, 如果达到最大使用次数，那么清理缓存
        self._partitions[partition_num][-1] += 1
        if self._partitions[partition_num][-1] >= self._partitions[partition_num][1]:
            del self._partitions[partition_num]
        return partitions

    def batch_sampler(self, num_workers: int, shuffle=False, batch_size=1):
        indices = []
        for _, row in self._path.iterrows():
            indices.append(list(range(row['lb'], row['ub'])))
        return PartitionSampler(indices=indices, num_workers=num_workers, shuffle=shuffle, batch_size=batch_size)

    def dataloader(self, num_workers: int, shuffle=False, batch_size=1, **kwargs) -> DataLoader:
        sampler = self.batch_sampler(shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        return DataLoader(
            self, collate_fn=self.collate_fn, batch_sampler=sampler, num_workers=num_workers, **kwargs)


class CSVDataset(PartitionDataset, ParserDataset):
    def __init__(self, path: pd.DataFrame, parser, func=None, **kwargs):
        super().__init__(path)
        self.parser = parser
        self.func = func
        self.kwargs = kwargs

    def read_file(self, path):
        return pd.read_csv(path, **self.kwargs)

    @staticmethod
    def get_partition_len(partition: pd.DataFrame):
        return partition.shape[0]

    def __getitem__(self, item) -> pd.DataFrame:
        partition_num = self._get_partition_num(item)
        # print(torch.utils.data.get_worker_info().id, partition_num)
        partition = self._get_partition(partition_num)
        df = partition.iloc[[item - self._path['lb'].iloc[partition_num]]]
        return df


class DFDataset(ParserDataset):
    def __init__(self, df: pd.DataFrame, parser, func=None):
        super().__init__(parser=parser, func=func)
        self._df = df

    def __len__(self):
        return self._df.shape[0]

    def __getitem__(self, item) -> pd.DataFrame:
        return self._df.iloc[[item]]
