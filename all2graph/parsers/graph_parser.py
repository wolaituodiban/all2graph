from typing import Dict, List

import dgl
import numpy as np
import pandas as pd

from ..graph import RawGraph, Graph
from ..info import MetaInfo
from ..meta_struct import MetaStruct
from ..stats import ECDF
from ..globals import *


class GraphParser(MetaStruct):
    def __init__(
            self, dictionary: Dict[str, int], numbers: Dict[str, ECDF], tokenizer=None, scale_method=None,
            **scale_kwargs
    ):
        """

        Args:
            dictionary: 字典
            numbers: 数值型的分布
            tokenizer: lcut
            scale_method: 归一化方法
            scale_kwargs: 归一化的参数
        """
        super().__init__(initialized=True)
        self.dictionary = dictionary
        self.numbers = numbers
        self.tokenizer = tokenizer
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

    def encode(self, inputs: list) -> List[int]:
        output = [self.dictionary.get(str(x), self.default_code) for x in inputs]
        return output

    def decode(self, inputs):
        rd = {v: k for k, v in self.dictionary.items()}
        rd[self.default_code] = 'default'
        rd[self.mask_code] = 'mask'
        return [rd[x] for x in inputs]

    def scale(self, key, values) -> np.ndarray:
        """归一化"""
        if key not in self.numbers:
            return np.full_like(values, np.nan)
        elif self.scale_method == 'prob':
            return self.numbers[key].get_probs(values, **self.scale_kwargs)
        elif self.scale_method == 'minmax':
            return self.numbers[key].minmax_scale(values, **self.scale_kwargs)
        else:
            raise KeyError('unknown scale_method {}'.format(self.scale_method))

    def encode_keys(self, keys):
        import torch
        if self.tokenizer is None:
            return torch.tensor(self.encode(keys), dtype=torch.long).unsqueeze(dim=-1)
        else:
            keys = list(map(self.tokenizer.lcut, keys))
            max_len = max(map(len, keys))
            output = []
            for key in keys:
                if len(key) < max_len:
                    key += [None] * (max_len - len(key))
                key = self.encode(key)[VALUE]
                output.append(key)
            return torch.tensor(output, dtype=torch.long)

    def __call__(self, raw_graph: RawGraph) -> Graph:
        """

        :param raw_graph:
        :return:
        """
        import torch
        edges = torch.tensor(raw_graph.edges[0], dtype=torch.long), torch.tensor(raw_graph.edges[1], dtype=torch.long)
        formatted_values = raw_graph.formatted_values
        tokens = torch.tensor(self.encode(formatted_values), dtype=torch.long)
        numbers = pd.to_numeric(formatted_values, errors='coerce')
        indices = []
        for inds in raw_graph.indices:
            temp = {}
            for key, ids in inds.items():
                numbers[ids] = self.scale(key, numbers[ids])
                temp[key] = torch.tensor(ids, dtype=torch.long)
            indices.append(temp)
        numbers = torch.tensor(numbers, dtype=torch.float32)

        graph = dgl.graph(edges, num_nodes=raw_graph.num_nodes)
        graph.ndata[TOKEN] = tokens
        graph.ndata[NUMBER] = numbers

        keys = list(raw_graph.unique_keys)
        key_tensor = self.encode_keys(keys)
        key_mapper = {key: i for i, key in enumerate(keys)}
        graph = Graph(
            graph=graph, targets=raw_graph.targets, key_tensor=key_tensor, key_mapper=key_mapper, indices=indices)
        return graph

    def extra_repr(self) -> str:
        return 'num_tokens={}, num_numbers={}, scale_method={}, scale_kwargs={}'.format(
            self.num_tokens, self.num_numbers, self.scale_method, self.scale_kwargs
        )

    @classmethod
    def from_data(cls, meta_info: MetaInfo, scale_method='prob', scale_kwargs=None, **kwargs):
        """

        Args:
            meta_info:
            scale_method:
            scale_kwargs:
            kwargs: MetaInfo.dictionary的参数
        Returns:

        """
        return cls(dictionary=meta_info.dictionary(**kwargs), numbers=meta_info.numbers, scale_method=scale_method,
                   **(scale_kwargs or {}))
