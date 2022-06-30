from typing import Dict, Set

from ..meta_struct import MetaStruct
from ..stats import ECDF
from ..graph import RawGraph
from ..globals import *
from ..utils import tqdm


class MetaInfo(MetaStruct):
    def __init__(self, types: Set[str], num_samples: int, doc_freqs: Dict[str, float],
                 num_counts: Dict[str, int], num_ecdfs: Dict[str, ECDF], **kwargs):
        super().__init__(**kwargs)
        assert set(num_counts) == set(num_ecdfs)
        self.types = types
        self.num_samples = num_samples
        self.doc_freqs = doc_freqs
        self.num_counts = num_counts
        self.num_ecdfs = num_ecdfs

    @classmethod
    def from_data(cls, raw_graph: RawGraph, num_bins=None):
        node_df = raw_graph.node_df
        node_df['copy'] = node_df[STRING]

        str_counts_df = node_df.pivot_table(values='copy', index=SAMPLE, columns=STRING, aggfunc='count', fill_value=0)
        doc_freqs = ((str_counts_df > 0).sum() / raw_graph.num_samples).to_dict()

        num_counts = {}
        num_ecdfs = {}
        for key, sub_node_df in node_df[[TYPE, NUMBER]].groupby(TYPE):
            num_count = sub_node_df[NUMBER].count()
            if num_count > 0:
                num_counts[key] = num_count
                num_ecdfs[key] = ECDF.from_data(sub_node_df[NUMBER], num_bins=num_bins)
        return super().from_data(types=raw_graph.unique_types, num_samples=raw_graph.num_samples,
                                 doc_freqs=doc_freqs, num_counts=num_counts, num_ecdfs=num_ecdfs)

    @classmethod
    def batch(cls, meta_infos, num_bins=None, disable=True, postfix=None):
        keys = None
        num_samples = None
        doc_freqs = None
        num_counts = None
        num_ecdfs = None
        for meta_info in tqdm(meta_infos, disable=disable, postfix=postfix):
            if keys is None:
                keys = set(meta_info.types)
                num_samples = meta_info.num_samples
                doc_freqs = dict(meta_info.doc_freqs)
                num_counts = dict(meta_info.num_counts)
                num_ecdfs = dict(meta_info.num_ecdfs)
                continue
            keys = keys.union(meta_info.types)
            doc_freqs = {key: doc_freq * num_samples / (num_samples + meta_info.num_samples)
                         for key, doc_freq in doc_freqs.items()}
            for key, doc_freq in meta_info.doc_freqs.items():
                if key in doc_freqs:
                    doc_freqs[key] += doc_freq * meta_info.num_samples / (num_samples + meta_info.num_samples)
                else:
                    doc_freqs[key] = doc_freq * meta_info.num_samples / (num_samples + meta_info.num_samples)
            num_samples += meta_info.num_samples
            for key, num_count in meta_info.num_counts.items():
                if key in num_counts:
                    num_ecdfs[key] = ECDF.batch(
                        [num_ecdfs[key], meta_info.num_ecdfs[key]], weights=[num_counts[key], num_count],
                        num_bins=num_bins
                    )
                    num_counts[key] += num_count
                else:
                    num_ecdfs[key] = meta_info.num_ecdfs[key]
                    num_counts[key] = num_count
        return super().batch(meta_infos, types=keys, doc_freqs=doc_freqs, num_samples=num_samples,
                             num_counts=num_counts, num_ecdfs=num_ecdfs)

    @property
    def num_types(self):
        return len(self.types)

    @property
    def num_numbers(self):
        return len(self.num_ecdfs)

    def dictionary(self, min_df=0, max_df=1, tokenizer=None) -> Dict[str, int]:
        dictionary = [s for s, df in self.doc_freqs.items() if min_df <= df <= max_df]
        if tokenizer is None:
            dictionary = self.types.union(dictionary)
        else:
            for ntype in self.types:
                dictionary += tokenizer.lcut(ntype)
            dictionary = set(dictionary)
        return {k: i for i, k in enumerate(dictionary)}

    def extra_repr(self) -> str:
        return 'num_types={}, num_numbers={}, num_tokens={}'.format(self.num_types, self.num_numbers, len(self.doc_freqs))
