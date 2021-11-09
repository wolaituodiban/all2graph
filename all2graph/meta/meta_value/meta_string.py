import os
from multiprocessing import Pool
from typing import Dict, Union, Iterable

import numpy as np
import pandas as pd

from .meta_value import MetaValue
from ...globals import EPSILON
from ...stats import Discrete, ECDF
from ...utils import MpMapFuncWrapper, progress_wrapper


def term_count_ecdf_to_doc_freq(term_count: ECDF, inverse=False) -> float:
    doc_freq = 1 - term_count.get_probs(0)
    if inverse:
        if doc_freq == 0:
            doc_freq += EPSILON
        return np.log(1/(doc_freq + EPSILON))
    else:
        return doc_freq


class MetaString(MetaValue):
    """类别节点"""
    def __init__(self, term_count_ecdf: Dict[str, ECDF], term_freq_ecdf: Dict[str, ECDF], **kwargs):
        """

        :param.py term_count_ecdf:
        :param.py term_freq_ecdf:
        :param.py kwargs:
        """
        # assert len(term_count_ecdf) > 0, '频率分布函数不能为空'
        assert all(isinstance(value, str) for value in term_count_ecdf)
        assert set(term_count_ecdf) == set(term_freq_ecdf)
        super().__init__(**kwargs)
        self.term_count_ecdf = term_count_ecdf
        self.term_freq_ecdf = term_freq_ecdf

    def __eq__(self, other, debug=False):
        if not super().__eq__(other):
            if debug:
                print('super not equal')
            return False
        if self.term_count_ecdf != other.term_count_ecdf:
            if debug:
                print('term_count_ecdf not equal')
            return False
        if self.term_freq_ecdf != other.term_freq_ecdf:
            if debug:
                print('term_freq_ecdf not equal')
            return False
        return True

    def __iter__(self):
        return iter(self.term_count_ecdf)

    def __len__(self):
        return len(self.term_count_ecdf)

    def keys(self):
        return self.term_count_ecdf.keys()

    def doc_freq(self, item: Union[str, Iterable[str]] = None, inverse=False) -> Union[float, Dict[str, float]]:
        if item is None:
            items = self.keys()
        elif isinstance(item, str):
            items = [item]
        else:
            items = item
        doc_freq = {k: term_count_ecdf_to_doc_freq(self.term_count_ecdf[k], inverse=inverse) for k in items}
        if isinstance(item, str):
            return doc_freq[item]
        else:
            return doc_freq

    def tf_idf_ecdf(self, item: Union[str, Iterable[str]] = None) -> Union[ECDF, Dict[str, ECDF]]:
        idf = self.doc_freq(item, inverse=True)
        if isinstance(idf, dict):
            return {
                k: ECDF(
                    quantiles=self.term_freq_ecdf[k].quantiles / idf[k],
                    probs=self.term_freq_ecdf[k].probs,
                    initialized=True
                )
                for k in idf
            }
        else:
            return ECDF(
                quantiles=self.term_freq_ecdf[item].quantiles / idf,
                probs=self.term_freq_ecdf[item].probs,
                initialized=True
            )

    @property
    def max_str_len(self):
        return max(map(len, self.term_count_ecdf))

    def to_discrete(self) -> Discrete:
        return Discrete.from_ecdfs(self.term_count_ecdf)

    def to_json(self) -> dict:
        output = super().to_json()
        output['term_count_ecdf'] = {k: v.to_json() for k, v in self.term_count_ecdf.items()}
        output['term_freq_ecdf'] = {k: v.to_json() for k, v in self.term_freq_ecdf.items()}
        return output

    @classmethod
    def from_json(cls, obj):
        obj = dict(obj)
        obj['term_count_ecdf'] = {k: ECDF.from_json(v) for k, v in obj['term_count_ecdf'].items()}
        obj['term_freq_ecdf'] = {k: ECDF.from_json(v) for k, v in obj['term_freq_ecdf'].items()}
        return super().from_json(obj)

    @classmethod
    def from_data(cls, num_samples, sample_ids, values, progress_bar=False, postfix='constructing meta string',
                  num_bins=None):
        # # # # # 计算node count
        node_count_series = pd.value_counts(sample_ids)

        # # # # # 计算tf-idf
        df = pd.DataFrame({'id': sample_ids, 'term': values})
        df.index.name = 'term_count'
        term_count_df = df.reset_index().groupby(['id', 'term'], sort=False).count()
        term_count_df = term_count_df.reset_index()

        term_count_ecdf = {}
        term_freq_ecdf = {}
        term_count_groupby = term_count_df.groupby('term', sort=False)
        term_count_groupby = progress_wrapper(term_count_groupby, disable=not progress_bar, postfix=postfix)
        for value, count_df in term_count_groupby:
            term_count = count_df['term_count'].values
            term_freq = term_count / node_count_series[count_df.id].values
            doc_freq = count_df.shape[0] / num_samples
            if doc_freq < 1:
                old_term_count = term_count
                term_count = np.zeros(num_samples)
                term_count[:old_term_count.shape[0]] = old_term_count

                old_term_freq = term_freq
                term_freq = np.zeros(num_samples)
                term_freq[:old_term_freq.shape[0]] = old_term_freq
            term_count_ecdf[value] = ECDF.from_data(term_count, num_bins=num_bins)
            term_freq_ecdf[value] = ECDF.from_data(term_freq, num_bins=num_bins)

        return super().from_data(term_count_ecdf=term_count_ecdf, term_freq_ecdf=term_freq_ecdf)

    @classmethod
    def reduce(cls, structs, weights=None, progress_bar=False, postfix='reducing meta string',
               processes=0, chunksize=None, num_bins=None):
        if weights is None:
            weights = np.full(len(structs), 1 / len(structs))
        else:
            weights = np.array(weights) / sum(weights)

        # term_count_ecdf和tf_idf_ecdf
        term_weights = {}
        term_count_ecdfs = {}
        term_freq_ecdfs = {}
        for weight, struct in zip(weights, structs):
            for term in struct:
                if term in term_count_ecdfs:
                    term_weights[term].append(weight)
                    term_count_ecdfs[term].append(struct.term_count_ecdf[term])
                    term_freq_ecdfs[term].append(struct.term_freq_ecdf[term])
                else:
                    term_weights[term] = [weight]
                    term_count_ecdfs[term] = [struct.term_count_ecdf[term]]
                    term_freq_ecdfs[term] = [struct.term_freq_ecdf[term]]

        for term in term_count_ecdfs:
            weight_sum = sum(term_weights[term])
            if weight_sum < 1 - EPSILON:
                term_count_ecdfs[term].append(ECDF([0], [1], initialized=True))
                term_freq_ecdfs[term].append(ECDF([0], [1], initialized=True))
                term_weights[term].append(1 - weight_sum)

        assert len(term_count_ecdfs) == len(term_freq_ecdfs) == len(term_weights)

        if processes == 0:
            term_count_ecdf = {}
            term_freq_ecdf = {}
            for term in progress_wrapper(term_count_ecdfs.keys(), disable=not progress_bar, postfix=postfix):
                term_count_ecdf[term] = ECDF.reduce(
                    term_count_ecdfs[term], weights=term_weights[term], num_bins=num_bins)
                term_freq_ecdf[term] = ECDF.reduce(term_freq_ecdfs[term], weights=term_weights[term], num_bins=num_bins)
        else:
            terms = term_count_ecdfs.keys()
            term_count_ecdfs = [{'structs': term_count_ecdfs[term], 'weights': term_weights[term]} for term in terms]
            term_freq_ecdfs = [{'structs': term_freq_ecdfs[term], 'weights': term_weights[term]} for term in terms]

            reduce_wrapper = MpMapFuncWrapper(ECDF.reduce, num_bins=num_bins)

            with Pool(processes) as pool:
                if chunksize is None:
                    chunksize = int(np.ceil(len(terms)/(processes or os.cpu_count())))
                term_count_ecdf = {
                    term: ecdf for term, ecdf in progress_wrapper(
                        zip(terms, pool.imap(reduce_wrapper, term_count_ecdfs, chunksize=chunksize)),
                        disable=not progress_bar, postfix=postfix, total=len(term_count_ecdfs)
                    )
                }
                term_freq_ecdf = {
                    term: ecdf for term, ecdf in progress_wrapper(
                        zip(terms, pool.imap(reduce_wrapper, term_freq_ecdfs, chunksize=chunksize)),
                        disable=not progress_bar, postfix=postfix + ' phase 2', total=len(term_freq_ecdfs)
                    )
                }

        return super().reduce(
            structs, weights=weights, term_count_ecdf=term_count_ecdf, term_freq_ecdf=term_freq_ecdf)

    def extra_repr(self) -> str:
        return 'num_strings={}'.format(len(self))
