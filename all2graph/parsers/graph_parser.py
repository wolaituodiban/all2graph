from typing import Dict, List

import dgl
import numpy as np

from ..globals import *
from ..graph import RawGraph, Graph
from ..info import MetaInfo
from ..meta_struct import MetaStruct
from ..stats import ECDF


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
        return cls(dictionary=meta_info.dictionary(**kwargs), num_ecdfs=meta_info.num_ecdfs,
                   scale_method=scale_method, **(scale_kwargs or {}))

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
            # 分词
            keys = list(map(self.tokenizer.lcut, keys))
            max_len = max(map(len, keys))
            output = []
            for key in keys:
                # pre padding
                if len(key) < max_len:
                    key = [None] * (max_len - len(key)) + key
                # encoding
                key = self.encode(key)[VALUE]
                output.append(key)
            return torch.tensor(output, dtype=torch.long)

    def __call__(self, raw_graph: RawGraph) -> Graph:
        """

        :param raw_graph:
        :return:
        """
        import torch
        # 序列和图需要保证双向映射
        # 然而序列中包含图中不存在的padding
        # 所以在图中增加一个作为padding的孤立点

        # parsing edges
        edges = torch.tensor(raw_graph.edges[0], dtype=torch.long), torch.tensor(raw_graph.edges[1], dtype=torch.long)

        # parsing values

        graph = dgl.graph(edges, num_nodes=raw_graph.num_nodes)
        node_df = raw_graph.node_df.set_index(TYPE)
        for t in node_df.index.unique():
            node_df.loc[t, NUMBER] = self.scale(t, node_df.loc[t, NUMBER])
        graph.ndata[STRING] = torch.tensor(self.encode(node_df[STRING]), dtype=torch.long)
        graph.ndata[NUMBER] = torch.tensor(node_df[NUMBER].tolist(), dtype=torch.float32)

        # parsing type and targets
        unique_types = list(raw_graph.unique_types)
        type_string = self.encode_keys(unique_types)
        type_mapper = {t: i for i, t in enumerate(unique_types)}
        targets = {target: type_mapper[target] for target in raw_graph.targets}

        # parsing seq2node
        max_seq_len = raw_graph.max_seq_len
        seq_mapper = {}
        node2seq = []
        seq_type = []
        seq_sample = []
        for (sample, t), nodes in raw_graph.seqs.items():
            if len(nodes) < max_seq_len:
                # 序列和图需要保证双向映射
                # 然而序列中包含图中不存在的padding
                # 实验显示，如果padding同一个值，会导致backward很慢
                # 因此随机padding，并设置为负数，方便后续发现
                nodes = nodes+np.random.randint(low=-raw_graph.num_nodes, high=0, size=max_seq_len-len(nodes)).tolist()
                node2seq.append(nodes)
            else:
                node2seq.append(nodes)
            seq_mapper[(sample, t)] = len(seq_mapper)
            seq_type.append(type_mapper[t])
            seq_sample.append(sample)
        node2seq = torch.tensor(node2seq, dtype=torch.long)
        seq_type = torch.tensor(seq_type, dtype=torch.long)
        seq_sample = torch.tensor(seq_sample, dtype=torch.long)

        # parsing node2seq
        seq2node = []
        for sample, t, loc in raw_graph.nodes:
            seq2node.append((seq_mapper[sample, t], loc))
        graph.ndata[SEQ2NODE] = torch.tensor(seq2node, dtype=torch.long)
        return Graph(graph=graph, node2seq=node2seq, seq_type=seq_type, seq_sample=seq_sample, type_string=type_string,
                     targets=targets, readout=type_mapper[READOUT], type_mapper=type_mapper)

    def extra_repr(self) -> str:
        return 'num_tokens={}, num_numbers={}, scale_method={}, scale_kwargs={}'.format(
            self.num_tokens, self.num_numbers, self.scale_method, self.scale_kwargs
        )

