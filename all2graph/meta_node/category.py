import json
from typing import Dict, Optional, Hashable

import pandas as pd

from .meta_node import MetaNode
from ..stats import Discrete, ECDF


class Category(MetaNode):
    """类别节点"""
    def __init__(self, ecdfs: Dict[Optional[Hashable], ECDF], **kwargs):
        """

        :param ecdfs: 在sample的维度上，看待每个value的频率分布函数
        """
        super().__init__(**kwargs)
        assert len(ecdfs) > 0, '频率分布函数不能为空'
        assert len({ecdf.num_samples for ecdf in ecdfs.values()}) == 1, '样本数不一致'
        self.ecdfs = ecdfs

    def __getitem__(self, item):
        return self.ecdfs[item]

    def __eq__(self, other):
        return super().__eq__(other) and self.ecdfs == other.ecdfs

    @property
    def num_nodes(self) -> int:
        return sum(freq.mean for freq in self.ecdfs.values()) * self[list(self.ecdfs)[0]].num_samples

    def to_discrete(self) -> Discrete:
        # todo
        pass

    def to_json(self) -> dict:
        output = super().to_json()
        output['ecdfs'] = {k: v.to_json() for k, v in self.ecdfs.items()}
        return output

    @classmethod
    def from_json(cls, obj):
        if isinstance(obj, str):
            obj = json.loads(obj)
        else:
            obj = dict(obj)
        obj['ecdfs'] = {k: ECDF.from_json(v) for k, v in obj['ecdfs'].items()}
        return super().from_json(obj)

    @classmethod
    def from_data(cls, df: pd.DataFrame, id_col='id', value_col='value', **kwargs):
        """
        :params df: 数据
        :params id_col: 用来确定唯一样本的id的列名
        :params value_col: 类别的值的列名
        """
        ecdfs = {}
        count_df = df.reset_index().groupby([id_col, value_col]).count()
        num_samples = count_df.index.get_level_values(0).unique().shape[0]
        for value, count in count_df.groupby(level=value_col):
            freq = count.values[:, 0].tolist()
            if len(freq) < num_samples:
                freq += [0] * (num_samples - len(freq))
            ecdfs[value] = ECDF.from_data(freq)
        return super().from_data(df, ecdfs=ecdfs, **kwargs)

    @classmethod
    def merge(cls, cats, **kwargs):
        ecdfs = {}
        num_samples = 0
        for cat in cats:
            num_samples += cat[list(cat.ecdfs)[0]].num_samples
            for value, freq in cat.ecdfs.items():
                if value not in ecdfs:
                    ecdfs[value] = [freq]
                else:
                    ecdfs[value].append(freq)
        ecdfs = {k: ECDF.merge(v) for k, v in ecdfs.items()}
        # 将所有值的频率分布补0，直到样本数一致
        for value, freq in ecdfs.items():
            if freq.num_samples < num_samples:
                ecdfs[value] = ECDF.merge([freq, ECDF.from_data([0] * (num_samples-freq.num_samples), **kwargs)])
        return super().merge(cats, ecdfs=ecdfs, **kwargs)
