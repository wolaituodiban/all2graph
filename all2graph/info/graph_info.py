from typing import Dict, Set

from .meta_info import MetaInfo
from .node_info import NodeInfo
from .str_info import StrInfo
from ..stats import ECDF
from ..utils import tqdm


class GraphInfo(MetaInfo):
    def __init__(self, node_infos: Dict[str, NodeInfo], str_info: StrInfo, **kwargs):
        super().__init__(**kwargs)
        self.node_infos = node_infos
        self.str_info = str_info

    @classmethod
    def batch(cls, graph_infos, num_bins=None, fast=True, disable=False, postfix=None, **kwargs):
        num_samples = 0
        node_infos = None
        str_info = None
        for graph_info in tqdm(graph_infos, disable=disable, postfix=postfix):
            num_samples += graph_info.num_samples
            if node_infos is None:
                node_infos = dict(graph_info.node_infos)
                str_info = graph_info.str_info
                continue
            for ntype in set(node_infos).union(graph_info.node_infos):
                node_infos[ntype] = NodeInfo.batch(
                    [node_infos.get(ntype, NodeInfo.empty(num_samples)),
                     graph_info.node_infos.get(ntype, NodeInfo.empty(graph_info.num_samples))],
                    num_bins=num_bins, fast=fast
                )
            str_info = StrInfo.batch([str_info, graph_info.str_info], num_bins=num_bins)
        return super().batch(graph_infos, node_infos=node_infos, str_info=str_info)

    @property
    def numbers(self) -> Dict[str, ECDF]:
        return {key: info.num_ecdf for key, info in self.node_infos.items() if info.num_ecdf is not None}

    @property
    def keys(self) -> Set[str]:
        return set(self.node_infos.keys())

    @property
    def num_samples(self):
        return self.str_info.num_samples

    @property
    def num_types(self):
        return len(self.keys)

    @property
    def num_unique_str(self):
        return self.str_info.num_unique_str

    @property
    def num_num_keys(self):
        return len(self.numbers)

    @property
    def doc_freq(self) -> Dict[str, float]:
        return self.str_info.doc_freq

    @property
    def tf_idf(self) -> Dict[str, ECDF]:
        return self.str_info.tf_idf

    def dictionary(self, min_df=0, max_df=1, top_k=None, top_method='mean_tfidf', tokenizer=None) -> Dict[str, int]:
        dictionary = [k for k, df in self.doc_freq.items() if min_df <= df <= max_df]
        if top_k is not None:
            if top_method == 'max_tfidf':
                dictionary = [(k, v.max) for k, v in self.tf_idf.items() if k in dictionary]
            elif top_method == 'mean_tfidf':
                dictionary = [(k, v.mean) for k, v in self.tf_idf.items() if k in dictionary]
            elif top_method == 'max_tf':
                dictionary = [(k, v.max) for k, v in self.tf_idf.items() if k in dictionary]
            elif top_method == 'mean_tf':
                dictionary = [(k, v.mean) for k, v in self.str_info.freqs_ecdf.items() if k in dictionary]
            elif top_method == 'max_tc':
                dictionary = [(k, v.max) for k, v in self.str_info.freqs_ecdf.items() if k in dictionary]
            elif top_method == 'mean_tc':
                dictionary = [(k, v.mean) for k, v in self.str_info.freqs_ecdf.items() if k in dictionary]
            else:
                raise ValueError(
                    "top_method只能是('max_tfidf', 'mean_tfidf', 'max_tf', 'mean_tf', 'max_tc', mean_tc')其中之一"
                )
            dictionary = sorted(dictionary, key=lambda x: x[1])
            dictionary = [k[0] for k in dictionary[:top_k]]
        if tokenizer is None:
            dictionary = self.keys.union(dictionary)
        else:
            for ntype in self.keys:
                dictionary += tokenizer.lcut(ntype)
            dictionary = set(dictionary)
        return {k: i for i, k in enumerate(dictionary)}

    def extra_repr(self) -> str:
        return 'num_keys={}, num_unique_str={}, num_num_keys={}'.format(
            self.num_types, self.num_unique_str, self.num_num_keys)
