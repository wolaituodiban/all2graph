from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch

from ..graph import Graph
from ..macro import NULL, PRESERVED_WORDS, META
from ..meta_graph import MetaGraph
from ..utils.dgl_utils import dgl
from ..meta_struct import MetaStruct


class Transformer(MetaStruct):
    def __init__(self, number_range: Dict[str, Tuple[float, float]], strings: list,
                 names: List[str], segment_name=False):
        """
        Graph与dgl.DiGraph的转换器
        :param number_range: 数值分位数上界和下界
        :param strings: 字符串编码字典
        :param names: 如果是dict，那么dict的元素必须是list，代表name的分词
        """
        super().__init__(initialized=True)
        self.range_df = pd.DataFrame(number_range, index=['lower', 'upper']).T

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

    def encode(self, item) -> int:
        item = str(item).lower()
        if item in self.string_mapper:
            return self.string_mapper[item]
        else:
            return self.string_mapper[NULL]

    @classmethod
    def from_data(cls, meta_graph: MetaGraph, min_df=0, max_df=1, top_k=None, top_method='mean_tfidf',
                  lower=0.05, upper=0.95, segment_name=False):
        """

        :param meta_graph:
        :param min_df: 字符串最小文档平吕
        :param max_df: 字符串最大文档频率
        :param top_k: 选择前k个字符串
        :param top_method: 'max_tfidf', 'mean_tfidf', 'max_tf', 'mean_tf', 'max_tc', mean_tc'
        :param lower: 数值分位点下界
        :param upper: 数值分位点上界
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

        number_range = {
            key: (float(ecdf.value_ecdf.get_quantiles(lower)), float(ecdf.value_ecdf.get_quantiles(upper)))
            for key, ecdf in meta_graph.meta_numbers.items()
            if ecdf.value_ecdf.mean_var[1] > 0
        }

        names = list(meta_graph.meta_name)
        return cls(number_range=number_range, strings=strings, names=names, segment_name=segment_name)

    def _gen_dgl_meta_graph(
            self, component_ids: List[int], names: List[str], preds: List[int], succs: List[int]
    ) -> dgl.DGLGraph:
        if isinstance(self.names, dict):
            from ..json import JsonResolver
            resolver = JsonResolver(root_name=META, list_inner_degree=0)
            temps = [META] * len(names)
            aug_graph = Graph(component_ids=component_ids, names=temps, values=temps, preds=preds, succs=succs)
            for meta_node_id, name in enumerate(names):
                if name not in self.names:
                    continue
                resolver.insert_array(graph=aug_graph, component_id=-1, name=META, value=self.names[name],
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

        # 元图点特征
        graph.ndata['component_id'] = torch.tensor(component_ids, dtype=torch.long)
        graph.ndata['name'] = torch.tensor(list(map(self.encode, names)), dtype=torch.long)
        return graph

    def _gen_dgl_graph(self, component_ids: List[int], names: List[str], values: List[str], meta_node_ids: List[int],
                       preds: List[int], succs: List[int], meta_edge_ids: List[int]) -> dgl.DGLGraph:
        # 构造dgl graph
        graph = dgl.graph(
            data=(torch.tensor(preds, dtype=torch.long), torch.tensor(succs, dtype=torch.long)),
            num_nodes=len(component_ids)
        )

        # 图边特征
        graph.edata['meta_edge_id'] = torch.tensor(meta_edge_ids, dtype=torch.long)

        # 图点特征
        graph.ndata['meta_node_id'] = torch.tensor(meta_node_ids, dtype=torch.long)

        # 图数值特征
        number_df = pd.DataFrame({'number': values, 'name': names})
        number_df = number_df.merge(self.range_df, left_on='name', right_index=True, how='left')
        number_df['number'] = pd.to_numeric(number_df.number, errors='coerce')
        number_df['number'] = np.clip(number_df.number, number_df.lower, number_df.upper)
        number_df['number'] = (number_df['number'] - number_df.lower) / (number_df.upper - number_df.lower)
        graph.ndata['number'] = torch.tensor(number_df['number'].values, dtype=torch.float32)

        # 图字符特征
        graph.ndata['value'] = torch.tensor(list(map(self.encode, values)), dtype=torch.long)
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

        node_df = pd.DataFrame({k: v.numpy() for k, v in graph.ndata.items()})
        node_df['component_id'] = meta_graph.ndata['component_id'][node_df['meta_node_id']].numpy()
        if isinstance(self.names, dict):
            nx_meta_graph = dgl.to_networkx(meta_graph, node_attrs=['name', 'component_id'])
            name_mapper: Dict[int, str] = {}
            for node, node_attr in nx_meta_graph.nodes.items():
                if node_attr['component_id'] < 0:
                    continue
                name = reverse_string_mapper[int(node_attr['name'])]
                if name != META:
                    name_mapper[node] = name
                else:
                    succs = [succ for succ in nx_meta_graph.succ[node] if nx_meta_graph.nodes[succ]['component_id'] < 0]
                    succs_degree = nx_meta_graph.degree(succs)
                    succs_degree = sorted(succs_degree, key=lambda x: x[1], reverse=True)
                    name_mapper[node] = ''.join(
                        [reverse_string_mapper[int(nx_meta_graph.nodes[succ]['name'])] for succ, _ in succs_degree]
                    )
            node_df['name'] = node_df['meta_node_id'].map(name_mapper)
        else:
            node_df['name'] = meta_graph.ndata['name'][node_df['meta_node_id']].numpy()
            node_df['name'] = node_df['name'].map(self.reverse_string_mapper)

        node_df['value'] = node_df['value'].map(self.reverse_string_mapper)
        node_df = node_df.merge(self.range_df, left_on=['name'], right_index=True, how='left')
        node_df['number'] = (node_df['number'] + node_df['lower']) * (node_df['upper'] - node_df['lower'])
        node_df.loc[node_df['number'].notna(), 'value'] = node_df.loc[node_df['number'].notna(), 'number']

        preds, succs = graph.edges()
        return Graph(
            component_ids=node_df['component_id'].tolist(),
            names=node_df['name'].tolist(),
            values=node_df['value'].tolist(),
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
