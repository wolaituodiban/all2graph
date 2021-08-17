from ..meta_node import MetaNode
from ....stats import ECDF


class MetaNumber(MetaNode):
    def to_json(self) -> dict:
        output = super().to_json()
        output['meta_data'] = self.meta_data.to_json()
        return output

    @classmethod
    def from_json(cls, obj: dict):
        obj = dict(obj)
        obj['meta_data'] = ECDF.from_json(obj['meta_data'])
        return super().from_json(obj)

    @classmethod
    def from_data(cls, num_samples, sample_ids, values, **kwargs):
        meta_data = ECDF.from_data(values, **kwargs)
        return super().from_data(num_samples, sample_ids, values, meta_data=meta_data, **kwargs)

    @classmethod
    def reduce(cls, structs, weights=None, **kwargs):
        # meta data的weight可以从freq中推出
        meta_data = ECDF.reduce(
            [struct.meta_data for struct in structs],
            weights=[struct.freq.mean for struct in structs],
            **kwargs
        )
        return super().reduce(structs, meta_data=meta_data, weights=weights, **kwargs)
