from typing import Dict, Tuple, List

import dgl
import numpy as np
import pandas as pd
import torch

from ..graph import Graph
from ..globals import NULL, PRESERVED_WORDS, COMPONENT_ID, META_NODE_ID, META_EDGE_ID, VALUE, NUMBER, TYPE, META
from ..meta import MetaInfo, MetaNumber
from ..utils import Tokenizer
from ..meta_struct import MetaStruct


class GraphTranser(MetaStruct):
    def __init__(self, meta_numbers: Dict[str, MetaNumber], strings: list,
                 keys: List[str], tokenizer: Tokenizer = None):
        """
        Graph与dgl.DiGraph的转换器
        :param meta_numbers: 数值分布
        :param strings: 字符串编码字典
        :param keys: 如果是dict，那么dict的元素必须是list，代表name的分词
        """
        super().__init__(initialized=True)
        self.meta_numbers = meta_numbers
        self.keys = keys
        self.tokenizer = tokenizer
        all_words = PRESERVED_WORDS + strings
        if self.tokenizer is not None:
            for key in keys:
                all_words += self.tokenizer.lcut(key)
        else:
            all_words += keys
        self.string_mapper = {}
        for word in (word.lower() for word in all_words):
            if word not in self.string_mapper:
                self.string_mapper[word] = len(self.string_mapper)

        assert NULL in self.string_mapper
        assert set(map(type, self.string_mapper)) == {str}
        assert len(self.string_mapper) == len(set(self.string_mapper.values())), (
            len(self.string_mapper), len(set(self.string_mapper.values()))
        )
        assert len(self.string_mapper) == max(list(self.string_mapper.values())) + 1

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
                  tokenizer: Tokenizer = None):
        """

        :param meta_info:
        :param min_df: 字符串最小文档平吕
        :param max_df: 字符串最大文档频率
        :param top_k: 选择前k个字符串
        :param top_method: 'max_tfidf', 'mean_tfidf', 'max_tf', 'mean_tf', 'max_tc', mean_tc'
        :param tokenizer:
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
        return cls(keys=all_keys, meta_numbers=meta_numbers, strings=strings, tokenizer=tokenizer)

    def _gen_dgl_meta_graph(
            self, component_ids: List[int], keys: List[str], srcs: List[int], dsts: List[int], types: List[str]
    ) -> dgl.DGLGraph:
        # 构造dgl meta graph
        graph = dgl.graph(
            data=(torch.tensor(srcs, dtype=torch.long), torch.tensor(dsts, dtype=torch.long)),
            num_nodes=len(component_ids),
        )
        graph.edata[META_EDGE_ID] = torch.arange(0, graph.num_edges(), 1, dtype=torch.long)

        # 元图点特征
        graph.ndata[COMPONENT_ID] = torch.tensor(component_ids, dtype=torch.long)
        graph.ndata[META_NODE_ID] = torch.arange(0, len(component_ids), 1, dtype=torch.long)
        graph.ndata[VALUE] = torch.tensor(list(map(self.encode, keys)), dtype=torch.long)
        graph.ndata[TYPE] = torch.tensor(list(map(self.encode, types)), dtype=torch.long)
        return graph

    def _gen_dgl_graph(self, keys: List[str], values: List[str], meta_node_ids: List[int], types: List[str],
                       srcs: List[int], dsts: List[int], meta_edge_ids: List[int]) -> dgl.DGLGraph:
        # 构造dgl graph
        graph = dgl.graph(
            data=(torch.tensor(srcs, dtype=torch.long), torch.tensor(dsts, dtype=torch.long)),
            num_nodes=len(keys)
        )

        # 图边特征
        graph.edata[META_EDGE_ID] = torch.tensor(meta_edge_ids, dtype=torch.long)

        # 图点特征
        graph.ndata[META_NODE_ID] = torch.tensor(meta_node_ids, dtype=torch.long)

        # 图数值特征
        # 特殊情况：values = [[]]，此时需要先转成pandas.Series
        numbers = pd.to_numeric(pd.Series(values).values, errors='coerce')
        unique_names, inverse_indices = np.unique(keys, return_inverse=True)
        masks = np.eye(unique_names.shape[0], dtype=bool)[inverse_indices]
        for i in range(unique_names.shape[0]):
            numbers[masks[:, i]] = self.get_probs(unique_names[i], numbers[masks[:, i]])
        graph.ndata[NUMBER] = torch.tensor(numbers, dtype=torch.float32)

        # 图字符特征
        graph.ndata[VALUE] = torch.tensor(list(map(self.encode, values)), dtype=torch.long)
        graph.ndata[TYPE] = torch.tensor(list(map(self.encode, types)), dtype=torch.long)
        return graph

    def graph_to_dgl(self, graph: Graph) -> Tuple[dgl.DGLGraph, dgl.DGLGraph]:
        """

        :param graph:
        :return:
        """
        meta_node_ids, meta_node_id_mapper, meta_node_component_ids, meta_node_keys, meta_node_types \
            = graph.meta_node_info()
        meta_edge_ids, pred_meta_node_ids, succ_meta_node_ids = graph.meta_edge_info(meta_node_id_mapper)
        if self.tokenizer is not None:
            graph.segment_key(component_ids=meta_node_component_ids, keys=meta_node_keys, srcs=pred_meta_node_ids,
                              dsts=succ_meta_node_ids, types=meta_node_types, tokenizer=self.tokenizer)

        dgl_meta_graph = self._gen_dgl_meta_graph(
            component_ids=meta_node_component_ids, keys=meta_node_keys, srcs=pred_meta_node_ids,
            dsts=succ_meta_node_ids, types=meta_node_types)
        dgl_graph = self._gen_dgl_graph(
            keys=graph.key, values=graph.value, meta_node_ids=meta_node_ids, meta_edge_ids=meta_edge_ids,
            srcs=graph.src, dsts=graph.dst, types=graph.type)
        return dgl_meta_graph, dgl_graph

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
