from abc import abstractmethod
from multiprocessing import Pool
from typing import Tuple, List, Union, Iterable

import numpy as np
import pandas as pd

from ..graph import RawGraph
from ..meta_struct import MetaStruct
from ..info import MetaInfo
from ..utils import tqdm, iter_csv, mp_run


class DataParser(MetaStruct):
    def __init__(self, json_col, time_col, time_format, targets, add_self_loop, **kwargs):
        super().__init__(initialized=True, **kwargs)
        self.json_col = json_col
        self.time_col = time_col
        self.time_format = time_format
        self.targets = targets or []
        self.add_self_loop = add_self_loop

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

    def _add_readouts(self, graph: RawGraph):
        if self.targets:
            graph.add_readouts_(ntypes=self.targets, self_loop=self.add_self_loop)

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

    @abstractmethod
    def __call__(self, data: pd.DataFrame, disable: bool = True) -> RawGraph:
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def to_json(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_json(cls, obj: dict):
        raise NotImplementedError

    @classmethod
    def from_data(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def reduce(cls, structs, weights=None, **kwargs):
        raise NotImplementedError


class DataAugmenter(DataParser):
    def __init__(self, parsers: List[DataParser], weights: List[float] = None):
        super(DataParser, self).__init__(initialized=True)
        self.parsers = parsers
        if weights is None:
            self.weights = np.ones(len(self.parsers)) / len(self.parsers)
        else:
            self.weights = np.array(weights) / np.sum(weights)

    def __call__(self, data, disable: bool = True, **kwargs) -> Tuple[RawGraph, dict, List[dict]]:
        parser = np.random.choice(self.parsers, p=self.weights)
        return parser.__call__(data, disable=disable, **kwargs)
