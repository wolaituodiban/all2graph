from typing import Dict, List

import numpy as np
import pandas as pd

from ..graph import RawGraph, Graph
from ..info import MetaInfo
from ..meta_struct import MetaStruct
from ..stats import ECDF


class GraphParser(MetaStruct):
    def __init__(
            self, dictionary: Dict[str, int], numbers: Dict[str, ECDF], scale_method=None, **scale_kwargs
    ):
        """

        Args:
            dictionary: 字典
            numbers: 数值型的分布
            scale_method: 归一化方法
            scale_kwargs: 归一化的参数
        """
        super().__init__(initialized=True)
        self.dictionary = dictionary
        self.numbers = numbers
        self.scale_method = scale_method
        self.scale_kwargs = scale_kwargs

    @property
    def default_code(self):
        """token的默认编码"""
        return len(self.dictionary)

    @property
    def mask_code(self):
        return len(self.dictionary) + 1

    @property
    def num_tokens(self):
        return len(self.dictionary) + 2

    @property
    def num_numbers(self):
        return len(self.numbers)

    def encode_token(self, inputs: list) -> List[int]:
        output = [self.dictionary.get(str(x), self.default_code) for x in inputs]
        return output

    def scale(self, keys, values) -> np.ndarray:
        """归一化"""
        df = pd.DataFrame({'key': keys, 'number': pd.to_numeric(values, errors='coerce')})
        for key in df['key'].unique():
            mask = df['key'] == key
            if key not in self.numbers:
                df.loc[mask, 'number'] = np.full_like(df.loc[mask, 'number'], np.nan)
            elif self.scale_method == 'prob':
                df.loc[mask, 'number'] = self.numbers[key].get_probs(df.loc[mask, 'number'], **self.scale_kwargs)
            elif self.scale_method == 'minmax':
                df.loc[mask, 'number'] = self.numbers[key].minmax_scale(df.loc[mask, 'number'], **self.scale_kwargs)
            else:
                raise KeyError('unknown scale_method {}'.format(self.scale_method))
        return df['number'].values

    def __call__(self, graph: RawGraph) -> Graph:
        """

        :param graph:
        :return:
        """
        import torch
        edges = {k: (torch.tensor(u, dtype=torch.long), torch.tensor(v, dtype=torch.long))
                 for k, (u, v) in graph.edges.items()}
        sids = torch.tensor(graph.sids, dtype=torch.long)
        key_tokens = torch.tensor(self.encode_token(graph.keys), dtype=torch.long)
        key_of_values = graph.get_keys(range(graph.num_values))
        value_tokens = torch.tensor(self.encode_token(key_of_values), dtype=torch.long)
        numbers = torch.tensor(self.scale(key_of_values, graph.formated_values), dtype=torch.float32)
        return Graph.from_data(edges, sids=sids, key_tokens=key_tokens, value_tokens=value_tokens, numbers=numbers)

    def __eq__(self, other, debug=False):
        if not super().__eq__(other):
            if debug:
                print('super not equal')
            return False
        if self.dictionary != other.dictionary:
            if debug:
                print('dictionary not equal')
            return False
        if self.numbers != other.numbers:
            if debug:
                print('numbers not equal')
            return False
        if self.scale_method != other.scale_method:
            if debug:
                print('scale_method not equal')
            return False
        if self.scale_kwargs != other.scale_kwargs:
            if debug:
                print('scale_kwargs not equal')
            return False
        return True

    def to_json(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_json(cls, obj: dict):
        raise NotImplementedError

    @classmethod
    def reduce(cls, parsers: list, tokenizer=None, weights=None, num_bins=None, **kwargs):
        raise NotImplementedError

    def extra_repr(self) -> str:
        return 'num_tokens={}, num_numbers={}, scale_method={}, scale_kwargs={}'.format(
            self.num_tokens, self.num_numbers, self.scale_method, self.scale_kwargs
        )

    @classmethod
    def from_data(cls, meta_info: MetaInfo, scale_method='minmax', scale_kwargs=None, **kwargs):
        """

        Args:
            meta_info:
            scale_method:
            scale_kwargs:
            kwargs: MetaInfo.dictionary的参数
        Returns:

        """
        return cls(dictionary=meta_info.dictionary(**kwargs), numbers=meta_info.numbers,
                   scale_method=scale_method, **(scale_kwargs or {}))
