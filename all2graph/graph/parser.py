from typing import Dict, Tuple, List, Set

import dgl
import numpy as np
import pandas as pd
import torch

from ..graph import Graph
from ..globals import NULL, PRESERVED_WORDS, COMPONENT_ID, META_NODE_ID, META_EDGE_ID, VALUE, NUMBER, TYPE, META, \
    READOUT, KEY
from ..meta import MetaInfo, MetaNumber
from ..utils import Tokenizer, default_tokenizer
from ..meta_struct import MetaStruct


class GraphParser(MetaStruct):
    def __init__(self, meta_numbers: Dict[str, MetaNumber], strings: list, keys: List[str],
                 edge_type: Set[Tuple[str, str]], targets: List[str] = None, tokenizer: Tokenizer = default_tokenizer,
                 meta_mode=True):
        """
        Graph与dgl.DiGraph的转换器
        :param meta_numbers: 数值分布
        :param strings: 字符串编码字典
        :param keys: 如果是dict，那么dict的元素必须是list，代表name的分词
        :param edge_type:
        :param targets:
        :parma meta_mode: 如果是True，那么graph_to_dgl会生成一个元图和一个值图，否则只生成一个值图
        """
        super().__init__(initialized=True)
        self.meta_numbers = meta_numbers
        self.tokenizer = tokenizer
        self.targets = list(targets or [])
        self.key_mapper = {k: i for i, k in enumerate(keys + self.targets)}

        self.etype_mapper = {t: i for i, t in enumerate(edge_type)}
        for target in self.targets:
            if (READOUT, target) not in self.etype_mapper:
                self.etype_mapper[(READOUT, target)] = len(self.etype_mapper)

        all_words = PRESERVED_WORDS + strings
        for key in keys + self.targets:
            all_words += self.tokenizer.lcut(key)
        self.string_mapper = {}
        for word in (word.lower() for word in all_words):
            if word not in self.string_mapper:
                self.string_mapper[word] = len(self.string_mapper)

        assert all(i == self.string_mapper[w] for i, w in enumerate(PRESERVED_WORDS))
        assert set(map(type, self.string_mapper)) == {str}
        assert len(self.string_mapper) == len(set(self.string_mapper.values())), (
            len(self.string_mapper), len(set(self.string_mapper.values()))
        )
        assert len(self.string_mapper) == max(list(self.string_mapper.values())) + 1
        self.meta_mode = meta_mode

    def set_meta_mode(self, mode: bool):
        self.meta_mode = mode

    @property
    def num_strings(self):
        return len(self.string_mapper)

    @property
    def num_targets(self):
        return len(self.targets)

    @property
    def num_keys(self):
        return len(self.key_mapper)

    @property
    def num_numbers(self):
        return len(self.meta_numbers)

    @property
    def num_etypes(self):
        return len(self.etype_mapper)

    @property
    def reverse_string_mapper(self):
        return {v: k for k, v in self.string_mapper.items()}

    def get_quantiles(self, name, p, **kwargs):
        if name in self.meta_numbers:
            return self.meta_numbers[name].value_ecdf.get_quantiles(p, **kwargs)
        else:
            return np.full_like(p, np.nan)

    def get_probs(self, name, q, **kwargs):
        if name in self.meta_numbers:
            return self.meta_numbers[name].value_ecdf.get_probs(q, **kwargs)
        else:
            return np.full_like(q, np.nan)

    def encode(self, item) -> int:
        item = str(item).lower()
        if item in self.string_mapper:
            return self.string_mapper[item]
        else:
            return self.string_mapper[NULL]

    @classmethod
    def from_data(cls, meta_info: MetaInfo, min_df=0, max_df=1, top_k=None, top_method='mean_tfidf',
                  targets=None, tokenizer: Tokenizer = default_tokenizer, meta_mode=False):
        """

        :param meta_info:
        :param min_df: 字符串最小文档平吕
        :param max_df: 字符串最大文档频率
        :param top_k: 选择前k个字符串
        :param top_method: 'max_tfidf', 'mean_tfidf', 'max_tf', 'mean_tf', 'max_tc', mean_tc'
        :param targets:
        :param tokenizer:
        :param meta_mode: 如果True，那么graph_to_dgl会返回元图加值图，否则只返回值图
        """
        strings = [k for k, df in meta_info.meta_string.doc_freq().items() if min_df <= df <= max_df]
        if top_k is not None:
            if top_method == 'max_tfidf':
                strings = [(k, v.max) for k, v in meta_info.meta_string.tf_idf_ecdf(strings).items()]
            elif top_method == 'mean_tfidf':
                strings = [(k, v.mean) for k, v in meta_info.meta_string.tf_idf_ecdf(strings).items()]
            elif top_method == 'max_tf':
                strings = [(k, v.max) for k, v in meta_info.meta_string.term_freq_ecdf.items() if k in strings]
            elif top_method == 'mean_tf':
                strings = [(k, v.mean) for k, v in meta_info.meta_string.term_freq_ecdf.items() if k in strings]
            elif top_method == 'max_tc':
                strings = [(k, v.max) for k, v in meta_info.meta_string.term_count_ecdf.items() if k in strings]
            elif top_method == 'mean_tc':
                strings = [(k, v.mean) for k, v in meta_info.meta_string.term_count_ecdf.items() if k in strings]
            else:
                raise ValueError(
                    "top_method只能是('max_tfidf', 'mean_tfidf', 'max_tf', 'mean_tf', 'max_tc', mean_tc')其中之一"
                )
            strings = sorted(strings, key=lambda x: x[1])
            strings = [k[0] for k in strings[:top_k]]

        meta_numbers = {key: ecdf for key, ecdf in meta_info.meta_numbers.items() if ecdf.value_ecdf.mean_var[1] > 0}

        all_keys = list(meta_info.meta_name)
        return cls(keys=all_keys, meta_numbers=meta_numbers, strings=strings, tokenizer=tokenizer, targets=targets,
                   edge_type=meta_info.edge_type, meta_mode=meta_mode)

    def _graph_to_dgl(
            self, src: List[int], dst: List[int], value: List[str], number: bool, type: List[str] = None,
            key: List[str] = None, component_id: List[int] = None, meta_node_id: List[int] = None,
            meta_edge_id: List[int] = None, add_key=False, edge_type: List[Tuple[str, str]] = None
    ) -> dgl.DGLGraph:
        # 构造dgl graph
        graph = dgl.graph(
            data=(torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)),
            num_nodes=len(value)
        )
        graph.ndata[VALUE] = torch.tensor(list(map(self.encode, value)), dtype=torch.long)
        graph.ndata[TYPE] = torch.tensor(list(map(self.encode, type)), dtype=torch.long)

        if number:
            # 特殊情况：values = [[]]，此时需要先转成pandas.Series
            numbers = pd.to_numeric(pd.Series(value).values, errors='coerce')
            unique_names, inverse_indices = np.unique(key, return_inverse=True)
            masks = np.eye(unique_names.shape[0], dtype=bool)[inverse_indices]
            for i in range(unique_names.shape[0]):
                numbers[masks[:, i]] = self.get_probs(unique_names[i], numbers[masks[:, i]])
            graph.ndata[NUMBER] = torch.tensor(numbers, dtype=torch.float32)

        if component_id is not None:
            graph.ndata[COMPONENT_ID] = torch.tensor(component_id, dtype=torch.long)

        if meta_node_id is not None:
            graph.ndata[META_NODE_ID] = torch.tensor(meta_node_id, dtype=torch.long)

        if meta_edge_id is not None:
            graph.edata[META_EDGE_ID] = torch.tensor(meta_edge_id, dtype=torch.long)

        if add_key:
            graph.ndata[KEY] = torch.tensor([self.key_mapper[k] for k in key], dtype=torch.long)

        if edge_type is not None:
            graph.edata[TYPE] = torch.tensor([self.etype_mapper[e] for e in edge_type], dtype=torch.long)

        return graph

    def graph_to_dgl(self, graph: Graph):
        """

        :param graph:
        :return:
        """
        graph = graph.add_targets(self.targets)
        if self.meta_mode:
            meta_graph, meta_node_id, meta_edge_id = graph.meta_graph(self.tokenizer)
            dgl_meta_graph = self._graph_to_dgl(
                src=meta_graph.src, dst=meta_graph.dst, value=meta_graph.value, number=False, type=meta_graph.type,
                component_id=meta_graph.component_id)
            dgl_graph = self._graph_to_dgl(
                src=graph.src, dst=graph.dst, value=graph.value, key=graph.key, type=graph.type,
                meta_node_id=meta_node_id, meta_edge_id=meta_edge_id, number=True)
            return dgl_meta_graph, dgl_graph
        else:
            return self._graph_to_dgl(
                src=graph.src, dst=graph.dst, value=graph.value, number=True, key=graph.key, type=graph.type,
                component_id=graph.component_id, add_key=True, edge_type=graph.edge_type)

    __call__ = graph_to_dgl

    def graph_from_dgl(self, meta_graph: dgl.DGLGraph, graph: dgl.DGLGraph) -> Graph:
        reverse_string_mapper = self.reverse_string_mapper
        # 复原被拆分的keys
        # 此部分的逻辑需要参考Graph.segment_keys
        meta_node_keys = [reverse_string_mapper[int(n)] for n in meta_graph.ndata[VALUE]]
        meta_u, meta_v = meta_graph.edges()
        meta_node_mask = meta_graph.ndata[TYPE] == self.encode(META)  # 找到所有类型为META的元节点
        if meta_node_mask.sum() > 0:  # 如果没有META元节点，那么所有元节点的key都是未拆分的状态
            meta_edge_mask = meta_node_mask[meta_u]  # 找到所有前置节点为META的元边
            for v in range(len(meta_node_keys)):
                # 找到所有指向v，并且不是自连接的边，并且前置节点的类型是meta的边
                mask = (meta_u != v) & (meta_v == v) & meta_edge_mask
                if mask.sum() > 0:
                    meta_node_keys[v] = self.tokenizer.join([meta_node_keys[int(u)] for u in meta_u[mask]])
        keys = [meta_node_keys[i] for i in graph.ndata[META_NODE_ID]]

        # 复原numbers
        with graph.local_scope():
            numbers = graph.ndata[NUMBER].numpy()
            unique_names, inverse_indices = np.unique(keys, return_inverse=True)
            masks = np.eye(unique_names.shape[0], dtype=bool)[inverse_indices]
            for i in range(unique_names.shape[0]):
                numbers[masks[:, i]] = self.get_quantiles(unique_names[i], numbers[masks[:, i]])

        # 复原values
        values = pd.Series([reverse_string_mapper[value] for value in graph.ndata[VALUE].numpy()])
        notna = np.bitwise_not(np.isnan(numbers))
        values[notna] = numbers[notna]

        # 复原graph
        srcs, preds = graph.edges()
        return Graph(
            component_id=meta_graph.ndata[COMPONENT_ID][graph.ndata[META_NODE_ID]].numpy().tolist(),
            key=keys,
            value=values.tolist(),
            src=srcs.numpy().tolist(),
            dst=preds.numpy().tolist(),
            type=[reverse_string_mapper[int(n)] for n in graph.ndata[TYPE]]
        )

    def __eq__(self, other):
        raise NotImplementedError

    def to_json(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_json(cls, obj: dict):
        raise NotImplementedError

    @classmethod
    def reduce(cls, structs, weights=None, **kwargs):
        raise NotImplementedError

    def extra_repr(self) -> str:
        return 'num_numbers={}, num_strings={}, num_keys={}, targets={}, num_etype={}'.format(
            self.num_numbers, self.num_strings, self.num_keys, self.targets, self.num_etypes
        )
