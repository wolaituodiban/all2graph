import json
from typing import Dict, List

import numpy as np
import pandas as pd

from ..meta_node import MetaNode
from ....stats import Discrete, ECDF


class StringNode(MetaNode):
    """类别节点"""
    def __init__(self, node_freq, value_dist: Dict[str, ECDF], **kwargs):
        """

        :param value_dist: 在sample的维度上，看待每个value的频率分布函数
        """
        assert len(value_dist) > 0, '频率分布函数不能为空'
        assert len({ecdf.num_samples for ecdf in value_dist.values()}) == 1, '样本数不一致'
        assert all(isinstance(value, str) for value in value_dist)
        super().__init__(node_freq=node_freq, value_dist=value_dist, **kwargs)

    def __getitem__(self, item):
        return self.value_dist[item]

    def __len__(self):
        return len(self.value_dist)

    def __eq__(self, other):
        return super().__eq__(other) and self.value_dist == other.value_dist

    @property
    def max_len(self):
        return max(map(len, self.value_dist))

    def to_discrete(self) -> Discrete:
        return Discrete.from_ecdfs(self.value_dist)

    def to_json(self) -> dict:
        output = super().to_json()
        output[self.VALUE_DIST] = {k: v.to_json() for k, v in self.value_dist.items()}
        return output

    @classmethod
    def from_json(cls, obj):
        if isinstance(obj, str):
            obj = json.loads(obj)
        else:
            obj = dict(obj)
        obj[cls.VALUE_DIST] = {k: ECDF.from_json(v) for k, v in obj[cls.VALUE_DIST].items()}
        return super().from_json(obj)

    @classmethod
    def from_data(cls, num_samples, sample_ids, values, **kwargs):
        # 比较np.unique和pd.groupby的性能差异
        value_dists = {}
        id_col = 'id'
        value_col = 'value'
        df = pd.DataFrame({id_col: sample_ids, value_col: values})
        count_df = df.reset_index().groupby([id_col, value_col], sort=False).count()

        for value, count in count_df.groupby(level=value_col, sort=False):
            freq = count.values[:, 0]
            if freq.shape[0] < num_samples:
                old_freq = freq
                freq = np.zeros(num_samples)
                freq[:old_freq.shape[0]] = old_freq
            value_dists[value] = ECDF.from_data(freq)
        kwargs[cls.VALUE_DIST] = value_dists
        return super().from_data(num_samples=num_samples, sample_ids=sample_ids, values=values, **kwargs)

    @classmethod
    def reduce(cls, cats, **kwargs):
        value_dists: Dict[str, List[ECDF]] = {}
        num_samples = 0
        for cat in cats:
            num_samples += cat[list(cat.value_dist)[0]].num_samples
            for value, freq in cat.value_dist.items():
                if value not in value_dists:
                    value_dists[value] = [freq]
                else:
                    value_dists[value].append(freq)
        # 将所有值的频率分布补0，直到样本数一致
        for value in value_dists:
            temp_sum_samples = sum(freq.num_samples for freq in value_dists[value])
            if temp_sum_samples < num_samples:
                zero_ecdf = ECDF.from_data(np.zeros(num_samples - temp_sum_samples), **kwargs)
                # 如果需要进一步提升性能，可以主动调用构造函数
                # zero_ecdf = ECDF([0], [1], num_samples=num_samples-temp_sum_samples, initialized=True)
                value_dists[value].append(zero_ecdf)
            value_dists[value] = ECDF.reduce(value_dists[value], **kwargs)
        kwargs[cls.VALUE_DIST] = value_dists
        return super().reduce(cats, **kwargs)
