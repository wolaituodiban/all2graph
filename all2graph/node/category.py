import json
from typing import Dict
from .meta_node import MetaNode
from ..stats import Discrete, ECDF


class Category(MetaNode):
    FREQS = 'freqs'
    """类别节点"""
    def __init__(self, freqs: Dict[str, ECDF]):
        """

        :param freqs: 每个类型的频率分布函数
        """
        super().__init__()
        num_samples = {freq.num_samples for freq in freqs.values()}
        assert len(num_samples) == 1, '样本数不一致'
        self.freqs = freqs

    def to_discrete(self) -> Discrete:
        # todo
        pass

    def to_json(self) -> dict:
        output = super().to_json()
        output[self.FREQS] = {k: v.to_json() for k, v in self.freqs.items()}
        return output

    @classmethod
    def from_json(cls, obj):
        if isinstance(obj, str):
            obj = json.loads(obj)
        else:
            obj = dict(obj)
        obj[cls.FREQS] = {k: ECDF.from_json(v) for k, v in obj[cls.FREQS]}
        return super().from_json(obj)

    @classmethod
    def from_data(cls, **kwargs):
        # todo
        pass

    @classmethod
    def merge(cls, **kwargs):
        # todo
        pass
