import numpy as np
from all2graph.meta_node.meta_node import MetaNode
from all2graph.stats import ECDF


class BoolNode(MetaNode):
    def __init__(self, node_freq: ECDF, value_dist: float, **kwargs):
        """
        :params node_freq:
        :params value_dist: float, True的概率
        """
        super().__init__(node_freq=node_freq, value_dist=value_dist, **kwargs)

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
            value_dist = num_samples/new_num_samples*value_dist + struct.num_samples/new_num_samples*struct.value_dist
            num_samples = new_num_samples
        kwargs[cls.VALUE_DIST] = value_dist
        return super().reduce(structs, **kwargs)
