import random
import math
from abc import abstractmethod
from typing import List, Tuple, Dict, Any
from gzip import GzipFile

import pandas as pd
import torch
from torch.utils.data import Dataset as _Dataset, DataLoader, IterableDataset

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
    """读取分片CSV的Dataset"""
    def __init__(self, path: pd.DataFrame, parser, func=None, **kwargs):
        """
        Args:
            path: dataframe, 长度等于样本数, 需要有一列path, 代表每一个样本对应的文件路径
            parser: 解析器, 实现一个__call__方法, 将df转换成模型输入, 同时需要实现一个get_targets方法, 将df装换成label
            func: dataframe预处理函数, 如果不是None,那么将在parser之前调用
            kwargs: pd.read_csv的额外参数
        """
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


class GzipGraphDataset(IterableDataset):
    def __init__(self, batch_size: int, num_samples: tuple):
        self.batch_size = batch_size
        self.num_samples = num_samples
            
    def __iter__(self):
        start = 0
        end = sum(n for _, n in self.num_samples)
        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # split workload
            per_worker = int(math.ceil(end/float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker
            
        return self._iter(start, end)
            
    def _iter(self, start, end):
        # 获取start到end的所有样本路径和下标
        all_samples = [[path, i] for path, n in self.num_samples for i in range(n)]
        all_samples = all_samples[start:end]
        
        # 将worker_samples转成dict格式，key是路径，value下标
        samples_dict = {}
        for path, i in all_samples:
            if path not in samples_dict:
                samples_dict[path] = [i]
            else:
                samples_dict[path].append(i)
        
        # shuffle path
        for path, indices in random.sample(samples_dict.items(), k=len(samples_dict)):
            with GzipFile(path, 'rb') as myzip:
                graph, label = torch.load(myzip)
            
            # shuffle index
            random.shuffle(indices)
            
            # mini-batch
            i = 0
            while i < len(indices):
                # 排序batch_ids, 防止label对不齐
                batch_ids = sorted(indices[i:i+self.batch_size])
                i += self.batch_size
                batch_graph = graph.sample_subgraph(batch_ids)
                batch_graph.events
                batch_graph.survival_times
                batch_graph.edge_feats
                yield batch_graph, {k: v[batch_ids] for k, v in label.items()}
                
    def dataloader(self, num_workers: int, **kwargs) -> DataLoader:
        return DataLoader(self, num_workers=num_workers, batch_size=None, **kwargs)