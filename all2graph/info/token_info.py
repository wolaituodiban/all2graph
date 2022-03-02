import numpy as np

from ..globals import EPSILON
from ..meta_struct import MetaStruct
from ..stats import ECDF


class TokenInfo(MetaStruct):
    def __init__(self, count: ECDF, freq: ECDF, **kwargs):
        super().__init__(**kwargs)
        self.count = count
        self.freq = freq

    def __eq__(self, other, debug=False):
        if not super().__eq__(other):
            if debug:
                print('super not equal')
            return False
        if self.count != other.count:
            if debug:
                print('count not equal')
            return False
        if self.freq != other.freq:
            if debug:
                print('freq not equal')
            return False
        return True

    @property
    def doc_freq(self):
        return 1 - self.count.get_probs(0)

    @property
    def idf(self):
        doc_freq = self.doc_freq
        if doc_freq == 0:
            doc_freq += EPSILON
        return np.log(1 / (doc_freq + EPSILON))

    @property
    def tf_idf(self):
        return ECDF(
            quantiles=self.freq.quantiles / self.idf,
            probs=self.freq.probs,
            initialized=True
        )

    def to_json(self) -> dict:
        output = super().to_json()
        output['count'] = self.count.to_json()
        output['freq'] = self.freq.to_json()
        return output

    @classmethod
    def from_json(cls, obj):
        obj = dict(obj)
        obj['count'] = ECDF.from_json(obj['count'])
        obj['freq'] = ECDF.from_json(obj['count'])
        return super().from_json(obj)

    @classmethod
    def from_data(cls, counts, num_nodes, num_bins=None):
        """

        Args:
            counts: 在每个样本中出现的次数
            num_nodes: 每个样本的点的数量
            num_bins:

        Returns:

        """
        count = ECDF.from_data(counts, num_bins=num_bins)
        freq = ECDF.from_data(counts / num_nodes, num_bins=num_bins)
        return super().from_data(count=count, freq=freq)

    @classmethod
    def reduce(cls, structs, weights=None, num_bins=None):
        counts = []
        freqs = []
        for struct in structs:
            counts.append(struct.count)
            freqs.append(struct.freq)
        count = ECDF.reduce(counts, weights, num_bins=num_bins)
        freq = ECDF.reduce(freqs, weights, num_bins=num_bins)
        return super().reduce(structs, weights=weights, count=count, freq=freq)

    def extra_repr(self) -> str:
        return 'count={}\nfreq={}'.format(self.count, self.freq)


class _TokenReducer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, tokens: TokenInfo):
        return TokenInfo.reduce(tokens, **self.kwargs)
