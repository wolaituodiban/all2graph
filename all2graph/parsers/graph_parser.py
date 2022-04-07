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
            self, dictionary: Dict[str, int], num_ecdfs: Dict[str, ECDF], tokenizer=None, scale_method=None,
            **scale_kwargs
    ):
        """

        Args:
            dictionary: 字典
            num_ecdfs: 数值型的分布
            tokenizer: lcut
            scale_method: 归一化方法
            scale_kwargs: 归一化的参数
        """
        super().__init__(initialized=True)
        self.dictionary = dictionary
        self.num_ecdfs = num_ecdfs
        self.tokenizer = tokenizer
        self.scale_method = scale_method
        self.scale_kwargs = scale_kwargs

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
        return cls(dictionary=meta_info.dictionary(**kwargs), num_ecdfs=meta_info.num_ecdfs, scale_method=scale_method,
                   **(scale_kwargs or {}))

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
        return len(self.num_ecdfs)

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
        if key not in self.num_ecdfs:
            return np.full_like(values, np.nan)
        elif self.scale_method == 'prob':
            return self.num_ecdfs[key].get_probs(values, **self.scale_kwargs)
        elif self.scale_method == 'minmax':
            return self.num_ecdfs[key].minmax_scale(values, **self.scale_kwargs)
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
        graph = dgl.graph(edges, num_nodes=raw_graph.num_nodes)

        key_mapper = {key: i for i, key in enumerate(raw_graph.unique_keys)}
        key_tensor = self.encode_keys(key_mapper)

        node_df = raw_graph.node_df
        graph.ndata[STRING] = torch.tensor(self.encode(node_df[STRING]), dtype=torch.long)
        graph.ndata[NUMBER] = torch.tensor(node_df[NUMBER], dtype=torch.float32)
        graph.ndata[KEY] = torch.tensor(node_df[KEY].map(key_mapper), dtype=torch.long)

        graph = Graph(
            graph=graph, key_tensor=key_tensor, key_mapper=key_mapper, targets=raw_graph.targets,
            splits=raw_graph.splits)
        return graph

    def extra_repr(self) -> str:
        return 'num_tokens={}, num_numbers={}, scale_method={}, scale_kwargs={}'.format(
            self.num_tokens, self.num_numbers, self.scale_method, self.scale_kwargs
        )

