from typing import Dict, List

import numpy as np
import pandas as pd

from ..meta_struct import MetaStruct
from ..stats import ECDF
from ..globals import *


class StrInfo(MetaStruct):
    def __init__(self, num_samples: int, counts_ecdf: Dict[str, ECDF], freqs_ecdf: Dict[str, ECDF], **kwargs):
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.counts_ecdf = counts_ecdf
        self.freqs_ecdf = freqs_ecdf

    @classmethod
    def empty(cls, num_samples):
        return cls(num_samples=num_samples, counts_ecdf={}, freqs_ecdf={}, initialized=True)

    @classmethod
    def from_data(cls, num_samples, df, num_bins=None, **kwargs):
        # print(df)
        node_counts = pd.value_counts(df[SAMPLE])
        str_counts = df.pivot_table(index=SAMPLE, columns=VALUE, values='value_copy', aggfunc='count', fill_value=0)
        # print(2222222)
        # print(str_counts)
        counts_ecdf = {}
        freqs_ecdf = {}
        for s, col in str_counts.iteritems():
            if s == 'null':
                continue
            counts_ecdf[s] = ECDF.from_data(col, num_bins=num_bins)
            freqs_ecdf[s] = ECDF.from_data(col / node_counts, num_bins=num_bins)
        return super().from_data(num_samples=num_samples, counts_ecdf=counts_ecdf, freqs_ecdf=freqs_ecdf)

    @classmethod
    def batch(cls, str_infos, num_bins=None, **kwargs):
        num_samples = []
        counts_ecdf = []
        freqs_ecdf = []
        for info in str_infos:
            num_samples.append(info.num_samples)
            counts_ecdf.append(info.counts_ecdf)
            freqs_ecdf.append(info.freqs_ecdf)
        counts_ecdf = pd.DataFrame(counts_ecdf).fillna(ECDF.from_data([0]))
        freqs_ecdf = pd.DataFrame(freqs_ecdf).fillna(ECDF.from_data([0]))
        counts_ecdf = {s: ECDF.batch(col, weights=num_samples, num_bins=num_bins) for s, col in counts_ecdf.iteritems()}
        freqs_ecdf = {s: ECDF.batch(col, weights=num_samples, num_bins=num_bins) for s, col in freqs_ecdf.iteritems()}
        return super().from_data(num_samples=sum(num_samples), counts_ecdf=counts_ecdf, freqs_ecdf=freqs_ecdf)

    @property
    def num_unique_str(self):
        return len(self.counts_ecdf)

    @property
    def doc_freq(self):
        return {s: ecdf.get_probs(0) for s, ecdf in self.counts_ecdf.items()}

    @property
    def idf(self):
        return {s: np.log(1 / (df + EPSILON)) for s, df in self.doc_freq.items()}

    @property
    def tf_idf(self):
        idf = self.idf
        return {
            s: ECDF(quantiles=freq.quantiles / idf[s], probs=freq.probs) for s, freq in self.freqs_ecdf.items()
        }

    def extra_repr(self) -> str:
        return 'num_unique_str={}'.format(self.num_unique_str)
