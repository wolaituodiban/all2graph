from typing import Dict

import numpy as np
import pandas as pd

from ..meta_node import MetaNode
from ....stats import Discrete, ECDF


class MetaString(MetaNode):
    """类别节点"""
    def __init__(self, freq, meta_data: Dict[str, ECDF], **kwargs):
        """

        :param meta_data: 在sample的维度上，看待每个value的频率分布函数
        """
        assert len(meta_data) > 0, '频率分布函数不能为空'
        assert all(isinstance(value, str) for value in meta_data)
        super().__init__(freq=freq, meta_data=meta_data, **kwargs)

    def __iter__(self):
        return iter(self.meta_data)

    def __getitem__(self, item):
        return self.meta_data[item]

    def __len__(self):
        return len(self.meta_data)

    @property
    def max_str_len(self):
        return max(map(len, self.meta_data))

    def to_discrete(self) -> Discrete:
        return Discrete.from_ecdfs(self.meta_data)

    def to_json(self) -> dict:
        output = super().to_json()
        output['meta_data'] = {k: v.to_json() for k, v in self.meta_data.items()}
        return output

    @classmethod
    def from_json(cls, obj):
        obj = dict(obj)
        obj['meta_data'] = {k: ECDF.from_json(v) for k, v in obj['meta_data'].items()}
        return super().from_json(obj)

    @classmethod
    def from_data(cls, num_samples, sample_ids, values, **kwargs):
        meta_data = {}
        df = pd.DataFrame({'id': sample_ids, 'value': values})
        count_df = df.reset_index().groupby(['id', 'value'], sort=False).count()

        for value, count in count_df.groupby(level='value', sort=False):
            freq = count.values[:, 0]
            if freq.shape[0] < num_samples:
                old_freq = freq
                freq = np.zeros(num_samples)
                freq[:old_freq.shape[0]] = old_freq
            meta_data[value] = ECDF.from_data(freq, **kwargs)
        return super().from_data(
            num_samples=num_samples, sample_ids=sample_ids, values=values, meta_data=meta_data, **kwargs
        )

    @classmethod
    def reduce(cls, structs, weights=None, **kwargs):
        if weights is None:
            weights = np.full(len(structs), 1 / len(structs))
        else:
            weights = np.array(weights) / sum(weights)

        meta_data_weights = np.array([weight * struct.freq.mean for weight, struct in zip(weights, structs)])
        meta_data_weights /= sum(meta_data_weights)

        all_strings = np.unique(np.concatenate([list(struct.meta_data.keys()) for struct in structs]))
        meta_data = {value: ([], []) for value in all_strings}
        for weight, struct in zip(weights, structs):
            for value in struct:
                meta_data[value][0].append(struct[value])
                meta_data[value][1].append(weight)

        for value in meta_data:
            weight_sum = sum(meta_data[value][1])
            if weight_sum < 1:
                meta_data[value][0].append(ECDF([0], [1], initialized=True))
                meta_data[value][1].append(1 - weight_sum)

        meta_data = {value: ECDF.reduce(ecdfs, weights=ws, **kwargs) for value, (ecdfs, ws) in meta_data.items()}

        return super().reduce(structs, weights=weights, meta_data=meta_data, **kwargs)
