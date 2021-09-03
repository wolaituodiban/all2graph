from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch

from ..graph import Graph
from ..macro import NULL, PRESERVED_WORDS, META, COMPONENT_IDS, META_NODE_IDS, META_EDGE_IDS, NAMES, VALUES, NUMBERS
from ..meta import MetaGraph, MetaNumber

from ..utils.dgl_utils import dgl
from ..meta_struct import MetaStruct


class GraphTranser(MetaStruct):
    def __init__(self, meta_numbers: Dict[str, MetaNumber], strings: list,
                 names: List[str], segment_name=False):
        """
        Graph与dgl.DiGraph的转换器
        :param meta_numbers: 数值分布
        :param strings: 字符串编码字典
        :param names: 如果是dict，那么dict的元素必须是list，代表name的分词
        """
        super().__init__(initialized=True)
        self.meta_numbers = meta_numbers

        all_words = PRESERVED_WORDS + strings
        if segment_name:
            import jieba
            self.names = {}
            for name in names:
                name_cut = jieba.lcut(name)
                self.names[name] = name_cut
                for word in name_cut:
                    all_words.append(word)
        else:
            self.names = names
            all_words += self.names

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
    def from_data(cls, meta_graph: MetaGraph, min_df=0, max_df=1, top_k=None, top_method='mean_tfidf',
                  segment_name=False):
        """

        :param meta_graph:
        :param min_df: 字符串最小文档平吕
        :param max_df: 字符串最大文档频率
        :param top_k: 选择前k个字符串
        :param top_method: 'max_tfidf', 'mean_tfidf', 'max_tf', 'mean_tf', 'max_tc', mean_tc'
        :param segment_name: 是否对name分词
        """
        strings = [k for k, df in meta_graph.meta_string.doc_freq().items() if min_df <= df <= max_df]
        if top_k is not None:
            if top_method == 'max_tfidf':
                strings = [(k, v.max) for k, v in meta_graph.meta_string.tf_idf_ecdf(strings).items()]
            elif top_method == 'mean_tfidf':
                strings = [(k, v.mean) for k, v in meta_graph.meta_string.tf_idf_ecdf(strings).items()]
            elif top_method == 'max_tf':
                strings = [(k, v.max) for k, v in meta_graph.meta_string.term_freq_ecdf.items() if k in strings]
            elif top_method == 'mean_tf':
                strings = [(k, v.mean) for k, v in meta_graph.meta_string.term_freq_ecdf.items() if k in strings]
            elif top_method == 'max_tc':
                strings = [(k, v.max) for k, v in meta_graph.meta_string.term_count_ecdf.items() if k in strings]
            elif top_method == 'mean_tc':
                strings = [(k, v.mean) for k, v in meta_graph.meta_string.term_count_ecdf.items() if k in strings]
            else:
                raise ValueError(
                    "top_method只能是('max_tfidf', 'mean_tfidf', 'max_tf', 'mean_tf', 'max_tc', mean_tc')其中之一"
                )
            strings = sorted(strings, key=lambda x: x[1])
            strings = [k[0] for k in strings[:top_k]]

        meta_numbers = {key: ecdf for key, ecdf in meta_graph.meta_numbers.items() if ecdf.value_ecdf.mean_var[1] > 0}

        names = list(meta_graph.meta_name)
        return cls(meta_numbers=meta_numbers, strings=strings, names=names, segment_name=segment_name)

    def _gen_dgl_meta_graph(
            self, component_ids: List[int], names: List[str], preds: List[int], succs: List[int]
    ) -> dgl.DGLGraph:
        if isinstance(self.names, dict):
            from ..json import JsonParser
            json_parser = JsonParser(root_name=META, list_inner_degree=0)
            temps = [META] * len(names)
            aug_graph = Graph(component_ids=component_ids, names=temps, values=temps, preds=preds, succs=succs)
            for meta_node_id, name in enumerate(names):
                if name not in self.names:
                    continue
                json_parser.insert_array(graph=aug_graph, component_id=-1, name=META, value=self.names[name],
                                         preds=[meta_node_id], local_index_mapper={}, global_index_mapper={})
            component_ids = aug_graph.component_ids
            names = aug_graph.values
            preds = aug_graph.preds
            succs = aug_graph.succs

        # 构造dgl meta graph
        graph = dgl.graph(
            data=(torch.tensor(preds, dtype=torch.long), torch.tensor(succs, dtype=torch.long)),
            num_nodes=len(component_ids),
        )
        graph.edata[META_EDGE_IDS] = torch.arange(0, graph.num_edges(), 1, dtype=torch.long)

        # 元图点特征
        graph.ndata[COMPONENT_IDS] = torch.tensor(component_ids, dtype=torch.long)
        graph.ndata[META_NODE_IDS] = torch.arange(0, len(component_ids), 1, dtype=torch.long)
        graph.ndata[NAMES] = torch.tensor(list(map(self.encode, names)), dtype=torch.long)
        return graph

    def _gen_dgl_graph(self, component_ids: List[int], names: List[str], values: List[str], meta_node_ids: List[int],
                       preds: List[int], succs: List[int], meta_edge_ids: List[int]) -> dgl.DGLGraph:
        # 构造dgl graph
        graph = dgl.graph(
            data=(torch.tensor(preds, dtype=torch.long), torch.tensor(succs, dtype=torch.long)),
            num_nodes=len(component_ids)
        )

        # 图边特征
        graph.edata[META_EDGE_IDS] = torch.tensor(meta_edge_ids, dtype=torch.long)

        # 图点特征
        graph.ndata[COMPONENT_IDS] = torch.tensor(component_ids, dtype=torch.long)
        graph.ndata[META_NODE_IDS] = torch.tensor(meta_node_ids, dtype=torch.long)

        # 图数值特征
        # 特殊情况：values = [[]]，此时需要先转成pandas.Series
        numbers = pd.to_numeric(pd.Series(values).values, errors='coerce')
        unique_names, inverse_indices = np.unique(names, return_inverse=True)
        masks = np.eye(unique_names.shape[0], dtype=bool)[inverse_indices]
        for i in range(unique_names.shape[0]):
            numbers[masks[:, i]] = self.get_probs(unique_names[i], numbers[masks[:, i]])
        graph.ndata[NUMBERS] = torch.tensor(numbers, dtype=torch.float32)

        # 图字符特征
        graph.ndata[VALUES] = torch.tensor(list(map(self.encode, values)), dtype=torch.long)
        return graph

    def graph_to_dgl(self, graph: Graph) -> Tuple[dgl.DGLGraph, dgl.DGLGraph]:
        """

        :param graph:
        :return:
        """
        meta_node_ids, meta_node_id_mapper, meta_node_component_ids, meta_node_names = graph.meta_node_info()
        meta_edge_ids, pred_meta_node_ids, succ_meta_node_ids = graph.meta_edge_info(meta_node_id_mapper)

        dgl_meta_graph = self._gen_dgl_meta_graph(component_ids=meta_node_component_ids, names=meta_node_names,
                                                  preds=pred_meta_node_ids, succs=succ_meta_node_ids)
        dgl_graph = self._gen_dgl_graph(
            component_ids=graph.component_ids, names=graph.names, values=graph.values, meta_node_ids=meta_node_ids,
            meta_edge_ids=meta_edge_ids, preds=graph.preds, succs=graph.succs
        )
        return dgl_meta_graph, dgl_graph

    __call__ = graph_to_dgl

    def graph_from_dgl(self, meta_graph: dgl.DGLGraph, graph: dgl.DGLGraph) -> Graph:
        reverse_string_mapper = self.reverse_string_mapper

        if isinstance(self.names, dict):
            nx_meta_graph = dgl.to_networkx(meta_graph, node_attrs=[NAMES, COMPONENT_IDS])
            name_mapper: Dict[int, str] = {}
            for node, node_attr in nx_meta_graph.nodes.items():
                if node_attr[COMPONENT_IDS] < 0:
                    continue
                name = reverse_string_mapper[int(node_attr[NAMES])]
                if name != META:
                    name_mapper[node] = name
                else:
                    succs = [succ for succ in nx_meta_graph.succ[node] if nx_meta_graph.nodes[succ][COMPONENT_IDS] < 0]
                    succs_degree = nx_meta_graph.degree(succs)
                    succs_degree = sorted(succs_degree, key=lambda x: x[1], reverse=True)
                    name_mapper[node] = ''.join(
                        [reverse_string_mapper[int(nx_meta_graph.nodes[succ][NAMES])] for succ, _ in succs_degree]
                    )
            names = [name_mapper[meta_node_id] for meta_node_id in graph.ndata[META_NODE_IDS].numpy()]
        else:
            names = meta_graph.ndata[NAMES][graph.ndata[META_NODE_IDS]]
            names = [reverse_string_mapper[name] for name in names.numpy()]

        with graph.local_scope():
            numbers = graph.ndata[NUMBERS].numpy()
            unique_names, inverse_indices = np.unique(names, return_inverse=True)
            masks = np.eye(unique_names.shape[0], dtype=bool)[inverse_indices]
            for i in range(unique_names.shape[0]):
                numbers[masks[:, i]] = self.get_quantiles(unique_names[i], numbers[masks[:, i]])

        values = pd.Series([reverse_string_mapper[value] for value in graph.ndata[VALUES].numpy()])
        notna = np.bitwise_not(np.isnan(numbers))
        values[notna] = numbers[notna]

        preds, succs = graph.edges()
        return Graph(
            component_ids=graph.ndata[COMPONENT_IDS].numpy().tolist(),
            names=names,
            values=values.tolist(),
            preds=preds.numpy().tolist(),
            succs=succs.numpy().tolist()
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
