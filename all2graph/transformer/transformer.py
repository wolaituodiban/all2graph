from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from ..graph import Graph
from ..macro import NULL
from ..meta_graph import MetaGraph
from ..utils.dgl_utils import dgl


class Transformer:
    def __init__(self, number_range: Dict[str, Tuple[float, float]], string_mapper: Dict[str, int],
                 name_segmentation: bool):
        """
        Graph与dgl.DiGraph的转换器
        :param number_range: 数值分位数上界和下界
        :param string_mapper: 字符串编码字典
        :param name_segmentation: 是否拆分name，默认使用jieba拆分
        """
        self.range_df = pd.DataFrame(number_range, index=['lower', 'upper']).T
        self.string_mapper = string_mapper
        self.name_segmentation = name_segmentation

    @property
    def reverse_string_mapper(self):
        return {v: k for k, v in self.string_mapper.items()}

    @classmethod
    def from_meta_graph(cls, meta_graph: MetaGraph, min_df=0, max_df=1, top_k=None, top_method='mean_tfidf',
                        lower=0.05, upper=0.95, name_segmentation=True):
        """

        :param meta_graph:
        :param min_df: 字符串最小文档平吕
        :param max_df: 字符串最大文档频率
        :param top_k: 选择前k个字符串
        :param top_method: 'max_tfidf', 'mean_tfidf', 'max_tf', 'mean_tf', 'max_tc', mean_tc'
        :param lower: 数值分位点下界
        :param upper: 数值分位点上界
        :param name_segmentation: 是否对name分词
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

        string_mapper = {k: i for i, k in enumerate(strings)}

        for name in meta_graph.meta_name:
            if name_segmentation:
                import jieba
                for key in jieba.cut(name):
                    key = key.lower()
                    if key not in string_mapper:
                        string_mapper[key] = len(string_mapper)
            elif name not in string_mapper:
                string_mapper[name] = len(string_mapper)

        if NULL not in string_mapper:
            string_mapper[NULL] = len(string_mapper)

        assert NULL in string_mapper
        assert len(string_mapper) == len(set(string_mapper.values())), (
            len(string_mapper), len(set(string_mapper.values()))
        )
        assert len(string_mapper) == max(list(string_mapper.values())) + 1

        number_range = {
            key: (float(ecdf.value_ecdf.get_quantiles(lower)), float(ecdf.value_ecdf.get_quantiles(upper)))
            for key, ecdf in meta_graph.meta_numbers.items()
            if ecdf.value_ecdf.mean_var[1] > 0
        }
        return cls(number_range=number_range, string_mapper=string_mapper, name_segmentation=name_segmentation)

    def _gen_dgl_meta_graph(self, meta_node_df: pd.DataFrame, meta_edge_df: pd.DataFrame) -> dgl.DGLGraph:
        # 构造dgl meta graph
        meta_edge_df = meta_edge_df.merge(
            meta_node_df, left_on=['component_id', 'pred_name'], right_on=['component_id', 'name'], how='left'
        )
        meta_edge_df = meta_edge_df.merge(
            meta_node_df, left_on=['component_id', 'succ_name'], right_on=['component_id', 'name'], how='left'
        )
        graph = dgl.graph(
            data=(
                torch.tensor(meta_edge_df['meta_node_id_x'].values, dtype=torch.long),
                torch.tensor(meta_edge_df['meta_node_id_y'].values, dtype=torch.long)
            ),
            num_nodes=meta_node_df.shape[0],
        )

        # 元图点特征
        graph.ndata['component_id'] = torch.tensor(meta_node_df['component_id'].values, dtype=torch.long)
        if self.name_segmentation:
            raise NotImplementedError
        else:
            graph.ndata['name'] = torch.tensor(
                meta_node_df['name'].map(self.string_mapper).fillna(self.string_mapper[NULL]).values,
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
            node_df['value'].apply(
                lambda x: self.string_mapper[x]
                if isinstance(x, str) and x in self.string_mapper
                else self.string_mapper[NULL]
            ).values,
            dtype=torch.long
        )
        return graph

    def graph_to_dgl(self, graph: Graph) -> Tuple[dgl.DGLGraph, dgl.DGLGraph]:
        """

        :param graph:
        :return:
        """
        node_df = graph.node_df()
        edge_df = graph.edge_df(node_df)
        meta_node_df = graph.meta_node_df(node_df)
        meta_edge_df = graph.meta_edge_df(node_df, edge_df)

        dgl_meta_graph = self._gen_dgl_meta_graph(meta_node_df, meta_edge_df)
        dgl_graph = self._gen_dgl_graph(
            node_df=node_df, edge_df=edge_df, meta_node_df=meta_node_df, meta_edge_df=meta_edge_df
        )
        return dgl_meta_graph, dgl_graph

    def graph_from_dgl(self, meta_graph: dgl.DGLGraph, graph: dgl.DGLGraph) -> Graph:
        node_df = pd.DataFrame(dict(graph.ndata))
        node_df['component_id'] = meta_graph.ndata['component_id'][node_df['meta_node_id']]

        node_df['name'] = meta_graph.ndata['name'][node_df['meta_node_id']]
        if self.name_segmentation:
            raise NotImplementedError
        else:
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


