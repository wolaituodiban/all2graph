import os
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
            data_col,
            time_col,
            time_format,
            targets,
            seq_keys: Set[Union[str, Tuple[str]]] = None,
            degree=0,
            r_degree=0,
            **kwargs):
        super().__init__(initialized=True, **kwargs)
        self.data_col = data_col
        self.time_col = time_col
        self.time_format = time_format
        self.targets = targets or []
        self.seq_keys = seq_keys
        self.degree = degree
        self.r_degree = r_degree

    def get_targets(self, df: pd.DataFrame):
        if self.targets:
            import torch
            return {
                k: torch.tensor(pd.to_numeric(df[k], errors='coerce').values, dtype=torch.float32)
                for k in self.targets if k in df
            }
        else:
            return {}

    def _analyse(self, inputs, **kwargs) -> Tuple[MetaInfo, pd.DataFrame]:
        df, path = inputs
        graph = self(df, disable=True)
        if path is not None:
            graph.save(path)
            df = df.copy()
            df['path'] = path
        meta_info = graph.meta_info(**kwargs)
        return meta_info, df.drop(columns=self.data_col)

    def analyse(self, data, dst=None, chunksize=64, disable=False, postfix='reading csv', processes=None, configs=None,
                **kwargs):
        """
        返回数据的元信息
        Args:
            data:
            dst: 如果不是None，那么在此文件夹下保存中间结果
            chunksize:
            disable:
            postfix:
            processes:
            configs:MetaInfo的额外参数
            **kwargs: read_csv参数

        Returns:

        """
        if dst is not None:
            assert not os.path.exists(dst), '{} already exists'.format(dst)
            os.mkdir(dst)
            data = (
                (df, os.path.join(dst, '{}.ag.rg'.format(i)))
                for i, df in enumerate(iter_csv(data, chunksize=chunksize, **kwargs))
            )
        else:
            data = ((df, None) for i, df in enumerate(iter_csv(data, chunksize=chunksize, **kwargs)))
        meta_infos: List[MetaInfo] = []
        dfs = []
        weights = []
        for meta_info, df in mp_run(
                self._analyse, data, kwds=configs or {}, processes=processes, disable=disable, postfix=postfix):
            meta_infos.append(meta_info)
            weights.append(df.shape[0])
            if dst:
                dfs.append(df)
        cls = meta_infos[0].__class__
        meta_info = cls.batch(meta_infos, weights=weights, disable=disable, processes=processes, **configs or {})
        if dst:
            df = pd.concat(dfs)
            df.to_csv(dst+'_path.csv', index=False)
            return meta_info, df
        return meta_info

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
