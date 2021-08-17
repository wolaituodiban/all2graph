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
        kwargs[cls.VALUE_DIST] = np.mean(values)
        return super().from_data(num_samples=num_samples, sample_ids=sample_ids, values=values, **kwargs)

    @classmethod
    def reduce(cls, structs, **kwargs):
        num_samples = 0
        value_dist = 0
        for struct in structs:
            new_num_samples = num_samples + struct.num_samples
            value_dist = num_samples/new_num_samples*value_dist + struct.num_samples/new_num_samples*struct.meta_data
            num_samples = new_num_samples
        kwargs[cls.VALUE_DIST] = value_dist
        return super().reduce(structs, **kwargs)
