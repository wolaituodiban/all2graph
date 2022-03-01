from abc import abstractmethod
from typing import List, Tuple, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset as _Dataset, DataLoader

from .sampler import PartitionSampler
from .utils import default_collate
from ..graph import Graph
from ..parsers import DataParser, GraphParser


class Dataset(_Dataset):
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def collate_fn(self, batches):
        raise NotImplementedError

    @abstractmethod
    def build_dataloader(self, num_workers: int, shuffle=False, batch_size=1, **kwargs) -> DataLoader:
        raise NotImplementedError


class ParserDataset(Dataset):
    def __init__(self, data_parser: DataParser, graph_parser: GraphParser):
        self.data_parser = data_parser
        self.graph_parser = graph_parser

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item) -> pd.DataFrame:
        raise NotImplementedError

    def collate_fn(self, batches: List[pd.DataFrame]) -> Tuple[Graph, Dict[str, torch.Tensor]]:
        df = pd.concat(batches)
        graph = self.data_parser(df, disable=True)
        graph = self.graph_parser(graph)
        label = self.data_parser.gen_targets(df)
        return graph, label

    def build_dataloader(self, **kwargs) -> DataLoader:
        return DataLoader(self, collate_fn=self.collate_fn, **kwargs)


class PartitionDataset(Dataset):
    def __init__(self, path: pd.DataFrame):
        """

        Args:
            path: 长度为样本数量，需要有一列path
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
            self._partitions = {
                partition_num: self.read_file(self._path.index[partition_num])
            }
        return self._partitions[partition_num]

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


class CSVDataset(PartitionDataset, ParserDataset):
    def __init__(self, path: pd.DataFrame, data_parser: DataParser, graph_parser: GraphParser, **kwargs):
        super().__init__(path)
        self.data_parser = data_parser
        self.graph_parser = graph_parser
        self.kwargs = kwargs

    def read_file(self, path):
        return pd.read_csv(path, **self.kwargs)


class DFDataset(ParserDataset):
    def __init__(self, df: pd.DataFrame, data_parser: DataParser, graph_parser: GraphParser):
        super().__init__(data_parser=data_parser, graph_parser=graph_parser)
        self._df = df

    def __len__(self):
        return self._df.shape[0]

    def __getitem__(self, item) -> pd.DataFrame:
        return self._df.iloc[[item]]


class GraphDataset(PartitionDataset):
    def read_file(self, path):
        return Graph.load(path)

    def __getitem__(self, item) -> Tuple[Graph, Dict[str, torch.Tensor]]:
        partition_num = self._get_partition_num(item)
        # print(torch.utils.data.get_worker_info().id, partition_num)
        graph, label = self._get_partition(partition_num)
        i = item - self._path['lb'].iloc[partition_num]
        graph = graph.component_subgraph(i)
        label = {k: v[i] for k, v in label.items()}
        return graph, label

    def collate_fn(self, batches: List[Tuple[Graph, Dict[str, torch.Tensor]]]) -> Tuple[Graph, Dict[str, torch.Tensor]]:
        graphs = []
        labels = []
        for graph, label in batches:
            graphs.append(graph)
            labels.append(label)
        return Graph.batch(graphs), default_collate(labels)
