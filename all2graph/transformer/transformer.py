from typing import Dict, Tuple

# import dgl
# import torch

from ..graph import Graph
from ..macro import NULL
from ..meta_graph import MetaGraph


class Transformer:
    def __init__(self, number_range: Dict[str, Tuple[float, float]], string_mapper: Dict[str, int], segmentation: bool):
        """
        Graph与dgl.DiGraph的转换器
        :param number_range: 数值分位数上界和下界
        :param string_mapper: 字符串编码字典
        :param segmentation: 是否拆分name，默认使用jieba拆分
        """
        self.number_range = number_range
        self.string_mapper = string_mapper
        self.segmentation = segmentation

    @classmethod
    def from_meta_graph(cls, meta_graph: MetaGraph, min_df=0, max_df=1, top_k=None, top_method='mean_tfidf',
                        lower=0.05, upper=0.95, segmentation=True):
        """

        :param meta_graph:
        :param min_df: 字符串最小文档平吕
        :param max_df: 字符串最大文档频率
        :param top_k: 选择前k个字符串
        :param top_method: 'max_tfidf', 'mean_tfidf', 'max_tf', 'mean_tf', 'max_tc', mean_tc'
        :param lower: 数值分位点下界
        :param upper: 数值分位点上界
        :param segmentation: 是否对name分词
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
            if segmentation:
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
        return cls(number_range=number_range, string_mapper=string_mapper, segmentation=segmentation)

    # def ag_graph_to_dgl_graph(self, graph: Graph) -> dgl.DGLGraph:
    #     # todo
    #     pass
    #
    # def dgl_graph_to_ag_graph(self, graph: dgl.DGLGraph) -> Graph:
    #     # todo
    #     pass
