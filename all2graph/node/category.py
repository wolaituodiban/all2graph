import json
from typing import Dict, Optional, Hashable

import pandas as pd

from .meta_node import MetaNode
from ..stats import Discrete, ECDF


class Category(MetaNode):
    FREQS = 'freqs'
    """类别节点"""
    def __init__(self, freqs: Dict[Optional[Hashable], ECDF], **kwargs):
        """

        :param freqs: 每个类型的频率分布函数
        """
        super().__init__(**kwargs)
        assert len(freqs) > 0, '频率分布函数不能为空'
        assert len({freq.num_samples for freq in freqs.values()}) == 1, '样本数不一致'
        self.freqs = freqs

    def __getitem__(self, item):
        return self.freqs[item]

    def __eq__(self, other):
        return super().__eq__(other) and self.freqs == other.freqs

    @property
    def num_samples(self):
        return self.freqs[list(self.freqs)[0]].num_samples

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
        obj[cls.FREQS] = {k: ECDF.from_json(v) for k, v in obj[cls.FREQS].items()}
        if 'null' in obj[cls.FREQS]:
            assert None not in obj[cls.FREQS]
            obj[cls.FREQS][None] = obj[cls.FREQS]['null']
            del obj[cls.FREQS]['null']
        return super().from_json(obj)

    @classmethod
    def from_data(cls, df: pd.DataFrame, id_col, value_col):
        """
        :params df: 数据
        :params id_col: 用来确定唯一样本的id的列名
        :params value_col: 类别的值的列名
        """
        num_samples = df[id_col].unique().shape[0]
        count_df = df.reset_index().groupby([id_col, value_col], dropna=False).count()
        freqs = {}
        na_freqs = []
        for value, count in count_df.groupby(level=value_col, dropna=False):
            freq = count.values[:, 0].tolist()
            if len(freq) < num_samples:
                freq += [0] * (num_samples - len(freq))
            if pd.notna(value) or value == 'null':
                freqs[value] = ECDF.from_data(freq)
            else:
                na_freqs.append(ECDF.from_data(freq))
        # 合并所有na的频率
        if len(na_freqs) > 0:
            freqs[None] = ECDF.merge(na_freqs)
        else:
            freqs[None] = ECDF(x=[0], y=[1], num_samples=num_samples)
        return cls(freqs)

    @classmethod
    def merge(cls, cats):
        freqs = {}
        num_samples = 0
        for cat in cats:
            num_samples += cat.num_samples
            for value, freq in cat.freqs.items():
                if value not in freqs:
                    freqs[value] = [freq]
                else:
                    freqs[value].append(freq)
        freqs = {k: ECDF.merge(v) for k, v in freqs.items()}
        # 将所有值的频率分布补0，直到样本数一致
        for value, freq in freqs.items():
            if freq.num_samples < num_samples:
                freqs[value] = ECDF.merge([freq, ECDF(x=[0], y=[1], num_samples=num_samples-freq.num_samples)])
        return cls(freqs=freqs)
