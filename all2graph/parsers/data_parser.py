from typing import List, Type

import numpy as np
import pandas as pd

from ..graph.raw_graph import RawGraph
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
            **kwargs):
        super().__init__(**kwargs)
        self.data_col = data_col
        self.time_col = time_col
        self.time_format = time_format
        self.targets = set(targets or set())

    def to_json(self) -> dict:
        outputs = super().to_json()
        outputs['data_col'] = self.data_col
        outputs['time_col'] = self.time_col
        outputs['time_format'] = self.time_format
        outputs['targets'] = list(self.targets)
        return outputs

    def get_targets(self, df: pd.DataFrame):
        if self.targets:
            import torch
            output = {}
            for k in self.targets:
                if k in df:
                    output[k] = torch.tensor(pd.to_numeric(df[k], errors='coerce').values, dtype=torch.float32) 
                else:
                    output[k] = torch.full((df.shape[0], ), fill_value=np.nan, dtype=torch.float32)
            return output
        else:
            return {}

    def _analyse(self, df, info_cls: Type[MetaInfo], **configs):
        graph = self(df, disable=True)
        info = info_cls.from_data(graph, **configs)
        return info

    def analyse(self, data, chunksize=64, disable=False, postfix='analysing', processes=None, info_cls=MetaInfo,
                configs=None, **kwargs):
        """
        返回数据的元信息
        Args:
            data:
            chunksize:
            disable:
            postfix:
            processes:
            configs:MetaInfo的额外参数
            **kwargs: read_csv参数

        Returns:

        """
        configs = configs or {}
        kwds = dict(configs)
        kwds['info_cls'] = info_cls
        data = iter_csv(data, chunksize=chunksize, **kwargs)
        infos = mp_run(self._analyse, data, kwds=kwds, processes=processes, disable=True)
        return info_cls.batch(infos, disable=disable, postfix=postfix, **configs)

    def __call__(self, data: pd.DataFrame, disable: bool = True) -> RawGraph:
        raise NotImplementedError

    def extra_repr(self) -> str:
        output = [
            'data_col="{}"'.format(self.data_col),
            'time_col="{}"'.format(self.time_col),
            'time_format={}'.format(None if self.time_format is None else '"{}"'.format(self.time_format)),
            'targets={}'.format(self.targets)
        ]
        return '\n'.join(output)


class DataAugmenter(DataParser):
    def __init__(self, parsers: List[DataParser], weights: List[float] = None):
        super(DataParser, self).__init__()
        self.parsers = parsers
        if weights is None:
            self.weights = np.ones(len(self.parsers)) / len(self.parsers)
        else:
            self.weights = np.array(weights) / np.sum(weights)

    def __call__(self, *args, **kwargs):
        parser = np.random.choice(self.parsers, p=self.weights)
        return parser.__call__(*args, **kwargs)
