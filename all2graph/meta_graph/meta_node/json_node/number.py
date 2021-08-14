import json
from ..meta_node import MetaNode
from ....stats import ECDF


class Number(MetaNode):
    def to_json(self) -> dict:
        output = super().to_json()
        output[self.NODE_FREQ] = self.node_freq.to_json()
        output[self.VALUE_DIST] = self.value_dist.to_json()
        return output

    @classmethod
    def from_json(cls, obj):
        if isinstance(obj, str):
            obj = json.loads(obj)
        else:
            obj = dict(obj)
        obj[cls.VALUE_DIST] = ECDF.from_json(obj[cls.VALUE_DIST])
        return super().from_json(obj)

    @classmethod
    def from_data(cls, num_samples, sample_ids, values, **kwargs):
        kwargs[cls.VALUE_DIST] = ECDF.from_data(values)
        return super().from_data(num_samples, sample_ids, values, **kwargs)

    @classmethod
    def reduce(cls, structs, **kwargs):
        node_freqs = []
        value_dists = []
        for struct in structs:
            node_freqs.append(struct.node_freq)
            value_dists.append(struct.value_dist)
        kwargs[cls.NODE_FREQ] = ECDF.reduce(node_freqs)
        kwargs[cls.VALUE_DIST] = ECDF.reduce(value_dists)
        return super().reduce(structs, **kwargs)