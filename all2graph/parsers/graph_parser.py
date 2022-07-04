from typing import Dict, List, Iterable

import numpy as np

from ..globals import *
from ..graph import Graph
from ..info import MetaInfo
from ..meta_struct import MetaStruct
from ..stats import ECDF


class GraphParser(MetaStruct):
    def __init__(
            self, dictionary: Dict[str, int], num_ecdfs: Dict[str, ECDF], tokenizer=None, scale_method=None,
            scale_kwargs=None, **kwargs
    ):
        """

        Args:
            dictionary: 字典
            num_ecdfs: 数值型的分布
            tokenizer: lcut
            scale_method: 归一化方法
            scale_kwargs: 归一化的参数
        """
        super().__init__(**kwargs)
        self.dictionary = dictionary
        self.num_ecdfs = num_ecdfs
        self.tokenizer = tokenizer
        self.scale_method = scale_method
        self.scale_kwargs = scale_kwargs or {}

    def to_json(self) -> dict:
        outputs = super().to_json()
        outputs['dictionary'] = self.dictionary
        outputs['num_ecdfs'] = {k: v.to_json() for k, v in self.num_ecdfs.items()}
        outputs['scale_method'] = self.scale_method
        outputs['scale_kwargs'] = self.scale_kwargs
        return outputs

    @classmethod
    def from_json(cls, obj: dict):
        obj = dict(obj)
        obj['num_ecdfs'] = {k: ECDF.from_json(v) for k, v in obj['num_ecdfs'].items()}
        return cls(**obj)

    @classmethod
    def from_data(cls, meta_info: MetaInfo, tokenizer=None, scale_method='prob', scale_kwargs=None, **kwargs):
        """

        Args:
            meta_info:
            tokenizer:
            scale_method:
            scale_kwargs:
            kwargs: MetaInfo.dictionary的参数
        Returns:

        """
        return cls(dictionary=meta_info.dictionary(tokenizer=tokenizer, **kwargs), num_ecdfs=meta_info.num_ecdfs,
                   tokenizer=tokenizer, scale_method=scale_method, scale_kwargs=scale_kwargs)

    @property
    def default_code(self):
        """token的默认编码"""
        return len(self.dictionary)

    @property
    def mask_code(self):
        return len(self.dictionary) + 1

    @property
    def padding_code(self):
        return len(self.dictionary) + 2

    @property
    def num_tokens(self):
        return len(self.dictionary) + 3

    @property
    def num_numbers(self):
        return len(self.num_ecdfs)

    def decode(self, inputs) -> np.ndarray:
        rd = {v: k for k, v in self.dictionary.items()}
        rd[self.default_code] = 'default'
        rd[self.mask_code] = 'mask'

        def decode(x):
            return rd[x]
        decode = np.vectorize(decode, otypes=[object])
        return decode(inputs)

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

    def encode(self, inputs: Iterable[str]) -> List[int]:
        default_code = self.default_code
        return [self.dictionary.get(x, default_code) for x in inputs]

    def encode_keys(self, keys: List[str]):
        import torch
        if self.tokenizer is None:
            return torch.tensor(self.encode(keys), dtype=torch.long).unsqueeze(dim=-1)
        else:
            # 分词
            keys = list(map(self.tokenizer.lcut, keys))
            max_len = max(map(len, keys))
            output = []
            for key in keys:
                # encoding
                key = self.encode(key)
                # pre padding
                if len(key) < max_len:
                    key = [self.padding_code] * (max_len - len(key)) + key
                output.append(key)
            return torch.tensor(output, dtype=torch.long)

    def call(self, raw_graph):
        """

        :param raw_graph:
        :return:
        """
        import dgl
        import torch
        # parsing edges
        edges = torch.tensor(raw_graph.edges[0], dtype=torch.long), torch.tensor(raw_graph.edges[1], dtype=torch.long)
        graph = dgl.graph(edges, num_nodes=raw_graph.num_nodes)

        # parsing ndata
        seq_info = raw_graph.seq_info()
        strings, numbers = raw_graph.formatted_values()
        for t, nodes in seq_info.type2node.items():
            numbers[nodes] = self.scale(t, numbers[nodes])
        graph.ndata[STRING] = torch.as_tensor(self.encode(strings), dtype=torch.long)
        graph.ndata[NUMBER] = torch.as_tensor(numbers, dtype=torch.float32)
        graph.ndata[SEQ2NODE] = torch.tensor(seq_info.seq2node, dtype=torch.long)

        # parsing type
        unique_types = list(raw_graph.unique_types)
        type_string = self.encode_keys(unique_types)
        type_mapper = {t: i for i, t in enumerate(unique_types)}

        # parsing seq
        seq_type = torch.tensor([type_mapper[t] for t in seq_info.seq_type], dtype=torch.long)
        seq_sample = torch.tensor(seq_info.seq_sample, dtype=torch.long)
        return Graph(graph=graph, seq_type=seq_type, seq_sample=seq_sample, type_string=type_string,
                     targets=raw_graph.targets, type_mapper=type_mapper)

    def __call__(self, *args, **kwds):
        return self.call(*args, **kwds)

    def extra_repr(self) -> str:
        return 'num_tokens={}, num_numbers={}, scale_method={}, scale_kwargs={}'.format(
            self.num_tokens, self.num_numbers, self.scale_method, self.scale_kwargs
        )
