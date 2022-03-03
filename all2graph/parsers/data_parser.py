from abc import abstractmethod
from inspect import ismethod
from typing import Tuple, List, Union, Set

import numpy as np
import pandas as pd

from ..graph import RawGraph
from ..info import MetaInfo
from ..meta_struct import MetaStruct
from ..utils import iter_csv, mp_run


class DataParser(MetaStruct):
    def __init__(
            self,
            json_col,
            time_col,
            time_format,
            targets,
            seq_keys: Set[Union[str, Tuple[str]]] = None,
            degree=0,
            r_degree=0,
            **kwargs):
        super().__init__(initialized=True, **kwargs)
        self.json_col = json_col
        self.time_col = time_col
        self.time_format = time_format
        self.targets = targets or []
        self.seq_keys = seq_keys
        self.degree = degree
        self.r_degree = r_degree

        # cache
        self.__configs = {}

    def get_targets(self, df: pd.DataFrame):
        if self.targets:
            import torch
            return {
                k: torch.tensor(pd.to_numeric(df[k], errors='coerce').values, dtype=torch.float32)
                for k in self.targets if k in df
            }
        else:
            return {}

    def _analyse(self, df: pd.DataFrame) -> Tuple[MetaInfo, int]:
        graph = self(df, disable=True)
        meta_info = graph.meta_info(**self.__configs)
        return meta_info, df.shape[0]

    def analyse(self, data, chunksize=64, disable=False, postfix='reading csv', processes=None, configs=None, **kwargs):
        """
        返回数据的元信息
        Args:
            data:
            chunksize:
            disable:
            postfix:
            processes:
            configs:
            **kwargs:

        Returns:

        """
        self.__configs = configs or {}

        meta_infos: List[MetaInfo] = []
        weights = []
        data = iter_csv(data, chunksize=chunksize, **kwargs)
        for meta_info, weight in mp_run(self._analyse, data, processes=processes, disable=disable, postfix=postfix):
            meta_infos.append(meta_info)
            weights.append(weight)
        cls = meta_infos[0].__class__
        return cls.reduce(meta_infos, weights=weights, disable=disable, processes=processes, **self.__configs)

    def _post_call(self, graph: RawGraph):
        if self.targets:
            graph.add_readouts_(ntypes=self.targets)
        if self.degree != 0 and self.r_degree != 0:
            graph.add_edges_by_key_(keys=self.seq_keys, degree=self.degree, r_degree=self.r_degree)

    @abstractmethod
    def __call__(self, data: pd.DataFrame, disable: bool = True) -> RawGraph:
        raise NotImplementedError

    def extra_repr(self) -> str:
        s = '\n,'.join(
            '{}={}'.format(k, v) for k, v in self.__dict__.items() if not ismethod(v) and not k.startswith('_')
        )
        return s


class DataAugmenter(DataParser):
    def __init__(self, parsers: List[DataParser], weights: List[float] = None):
        super(DataParser, self).__init__(initialized=True)
        self.parsers = parsers
        if weights is None:
            self.weights = np.ones(len(self.parsers)) / len(self.parsers)
        else:
            self.weights = np.array(weights) / np.sum(weights)

    def __call__(self, *args, **kwargs):
        parser = np.random.choice(self.parsers, p=self.weights)
        return parser.__call__(*args, **kwargs)
