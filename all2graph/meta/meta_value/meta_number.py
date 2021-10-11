import numpy as np
import pandas as pd

from .meta_value import MetaValue
from ...stats import ECDF


class MetaNumber(MetaValue):
    def __init__(self, count_ecdf: ECDF, value_ecdf: ECDF, **kwargs):
        """

        :param.py count_ecdf: 节点的计数分布
        :param.py value_ecdf: 节点的元数据
        :param.py kwargs:
        """
        super().__init__(**kwargs)
        self.count_ecdf = count_ecdf
        self.value_ecdf = value_ecdf

    def __eq__(self, other):
        return super().__eq__(other) \
               and self.count_ecdf == other.count_ecdf \
               and self.value_ecdf == other.value_ecdf

    def to_json(self) -> dict:
        output = super().to_json()
        output['count_ecdf'] = self.count_ecdf.to_json()
        output['value_ecdf'] = self.value_ecdf.to_json()
        return output

    @classmethod
    def from_json(cls, obj: dict):
        obj = dict(obj)
        obj['count_ecdf'] = ECDF.from_json(obj['count_ecdf'])
        obj['value_ecdf'] = ECDF.from_json(obj['value_ecdf'])
        return super().from_json(obj)

    @classmethod
    def from_data(cls, num_samples, sample_ids, values, num_bins=None):
        node_counts = pd.value_counts(sample_ids).values
        if node_counts.shape[0] < num_samples:
            old_node_counts = node_counts
            node_counts = np.zeros(num_samples)
            node_counts[:old_node_counts.shape[0]] = old_node_counts
        else:
            assert node_counts.shape[0] == num_samples
        count_ecdf = ECDF.from_data(node_counts, num_bins=num_bins)
        value_ecdf = ECDF.from_data(values, num_bins=num_bins)
        return super().from_data(count_ecdf=count_ecdf, value_ecdf=value_ecdf)

    @classmethod
    def reduce(cls, structs, weights=None, num_bins=None):
        if weights is None:
            weights = np.full(len(structs), 1 / len(structs))
        else:
            weights = np.array(weights) / sum(weights)

        count_ecdf = ECDF.reduce([struct.count_ecdf for struct in structs], weights=weights, num_bins=num_bins)
        # meta data的weight可以从freq中推出
        value_ecdf = ECDF.reduce(
            [struct.value_ecdf for struct in structs],
            weights=[w * struct.count_ecdf.mean for w, struct in zip(weights, structs)],
            num_bins=num_bins)
        return super().reduce(structs, count_ecdf=count_ecdf, value_ecdf=value_ecdf, weights=weights)

    def extra_repr(self) -> str:
        return 'count={}\nvalue={}'.format(self.count_ecdf, self.value_ecdf)
