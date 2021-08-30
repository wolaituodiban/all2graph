from functools import lru_cache
from typing import Dict, Tuple, List

import jieba
import numpy as np
import pandas as pd
import torch

from ..graph import Graph
from ..macro import NULL, PRESERVED_WORDS, META
from ..meta_graph import MetaGraph
from ..utils.dgl_utils import dgl


class Transformer:
    def __init__(self, number_range: Dict[str, Tuple[float, float]], strings: list,
                 names: List[str], segment_name=False):
        """
        Graph与dgl.DiGraph的转换器
        :param number_range: 数值分位数上界和下界
        :param strings: 字符串编码字典
        :param names: 如果是dict，那么dict的元素必须是list，代表name的分词
        """
        self.range_df = pd.DataFrame(number_range, index=['lower', 'upper']).T

        all_words = PRESERVED_WORDS + strings
        if segment_name:
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

    def encode(self, item):
        item = str(item).lower()
        if item in self.string_mapper:
            return self.string_mapper[item]
        else:
            return self.string_mapper[NULL]

    @classmethod
    def from_meta_graph(cls, meta_graph: MetaGraph, min_df=0, max_df=1, top_k=None, top_method='mean_tfidf',
                        lower=0.05, upper=0.95, segment_name=True):
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

    def _gen_dgl_meta_graph(self, meta_node_df: pd.DataFrame, meta_edge_df: pd.DataFrame) -> dgl.DGLGraph:
        meta_edge_df = meta_edge_df.merge(
            meta_node_df, left_on=['component_id', 'pred_name'], right_on=['component_id', 'name'], how='left'
        )
        meta_edge_df = meta_edge_df.merge(
            meta_node_df, left_on=['component_id', 'succ_name'], right_on=['component_id', 'name'], how='left'
        )

        if isinstance(self.names, dict):
            from ..json import JsonResolver
            resolver = JsonResolver(root_name=META, list_inner_degree=0)
            aug_graph = Graph(component_ids=meta_node_df['component_id'].tolist(), names=[META]*meta_node_df.shape[0],
                              values=[META]*meta_node_df.shape[0], preds=meta_edge_df['meta_node_id_x'].tolist(),
                              succs=meta_edge_df['meta_node_id_y'].tolist())
            for row in meta_node_df.itertuples():
                component_id, name, meta_node_id = row[1:]
                if name not in self.names:
                    continue
                resolver.insert_array(graph=aug_graph, component_id=-1, name=META, value=self.names[name],
                                      preds=[meta_node_id], local_index_mapper={}, global_index_mapper={})
            meta_node_df = aug_graph.node_df()
            meta_node_df['name'] = meta_node_df['value']
            meta_edge_df = aug_graph.edge_df(node_df=meta_node_df)
            meta_edge_df[['meta_node_id_x', 'meta_node_id_y']] = meta_edge_df[['pred', 'succ']]

        # 构造dgl meta graph
        graph = dgl.graph(
            data=(
                torch.tensor(meta_edge_df['meta_node_id_x'].values, dtype=torch.long),
                torch.tensor(meta_edge_df['meta_node_id_y'].values, dtype=torch.long)
            ),
            num_nodes=meta_node_df.shape[0],
        )

        # 元图点特征
        graph.ndata['component_id'] = torch.tensor(meta_node_df['component_id'].values, dtype=torch.long)
        graph.ndata['name'] = torch.tensor(
            meta_node_df['name'].apply(self.encode).values,
            dtype=torch.long
        )
        return graph

    def _gen_dgl_graph(self, node_df, edge_df, meta_node_df, meta_edge_df) -> dgl.DGLGraph:
        # 构造dgl graph
        graph = dgl.graph(
            data=(
                torch.tensor(edge_df.pred.values, dtype=torch.long),
                torch.tensor(edge_df.succ.values, dtype=torch.long)
            ),
            num_nodes=node_df.shape[0]
        )

        # 图边特征
        graph.edata['meta_edge_id'] = torch.tensor(
            edge_df.merge(
                meta_edge_df, on=['component_id', 'pred_name', 'succ_name'], how='left'
            )['meta_edge_id'].values,
            dtype=torch.long
        )

        # 图点特征
        graph.ndata['meta_node_id'] = torch.tensor(
            node_df.merge(meta_node_df, on=['component_id', 'name'], how='left')['meta_node_id'].values,
            dtype=torch.long
        )

        # 图数值特征
        node_df = node_df.merge(self.range_df, left_on='name', right_index=True, how='left')
        node_df['number'] = pd.to_numeric(node_df.value, errors='coerce')
        node_df['number'] = np.clip(node_df.number, node_df.lower, node_df.upper)
        node_df['number'] = (node_df['number'] - node_df.lower) / (node_df.upper - node_df.lower)
        graph.ndata['number'] = torch.tensor(node_df['number'].values, dtype=torch.float32)

        # 图字符特征
        graph.ndata['value'] = torch.tensor(
            node_df['value'].apply(self.encode).values,
            dtype=torch.long
        )
        return graph

    def graph_to_dgl(self, graph: Graph) -> Tuple[dgl.DGLGraph, dgl.DGLGraph]:
        """

        :param graph:
        :return:
        """
        node_df = graph.node_df()
        edge_df = graph.edge_df(node_df=node_df)
        meta_node_df = graph.meta_node_df(node_df=node_df)
        meta_edge_df = graph.meta_edge_df(node_df=node_df, edge_df=edge_df)

        dgl_meta_graph = self._gen_dgl_meta_graph(meta_node_df=meta_node_df, meta_edge_df=meta_edge_df)
        dgl_graph = self._gen_dgl_graph(
            node_df=node_df, edge_df=edge_df, meta_node_df=meta_node_df, meta_edge_df=meta_edge_df
        )
        return dgl_meta_graph, dgl_graph

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
