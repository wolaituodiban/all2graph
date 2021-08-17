import numpy as np
from ..meta_node import MetaNode
from ....stats import ECDF


class MetaBool(MetaNode):
    def __init__(self, freq: ECDF, meta_data: float, **kwargs):
        """
        :params node_freq:
        :params value_dist: float, True的概率
        """
        super().__init__(freq=freq, meta_data=meta_data, **kwargs)

    @classmethod
    def from_data(cls, num_samples, sample_ids, values, **kwargs):
        return super().from_data(
            num_samples=num_samples, sample_ids=sample_ids, values=values, meta_data=float(np.mean(values)), **kwargs
        )

    @classmethod
    def reduce(cls, structs, weights=None, **kwargs):
        if weights is None:
            weights = np.full(len(structs), 1 / len(structs))
        else:
            weights = np.array(weights) / sum(weights)
        meta_data = sum(w * struct.meta_data for w, struct in zip(weights, structs))
        return super().reduce(structs, meta_data=meta_data, weights=weights, **kwargs)
