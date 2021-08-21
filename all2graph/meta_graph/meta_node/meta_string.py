from typing import Dict

import numpy as np
import pandas as pd
from toad.utils.progress import Progress

from .meta_node import MetaNode
from ...stats import Discrete, ECDF


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

    def keys(self):
        return self.meta_data.keys()

    def values(self):
        return self.meta_data.values()

    def items(self):
        return self.meta_data.items()

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
    def from_data(cls, num_samples, sample_ids, values, progress_bar=False, suffix='constructing meta string',
                  **kwargs):
        meta_data = {}
        df = pd.DataFrame({'id': sample_ids, 'value': values})
        count_df = df.reset_index().groupby(['id', 'value'], sort=False).count()

        progress = count_df.groupby(level='value', sort=False)
        if progress_bar:
            progress = Progress(progress)
            progress.suffix = suffix
        for value, count in progress:
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
    def reduce(cls, structs, weights=None, progress_bar=False, suffix='reducing meta string', **kwargs):
        if weights is None:
            weights = np.full(len(structs), 1 / len(structs))
        else:
            weights = np.array(weights) / sum(weights)

        meta_data = {}
        meta_data_w = {}
        for weight, struct in zip(weights, structs):
            for value, freq in struct.items():
                if value in meta_data:
                    meta_data[value].append(freq)
                    meta_data_w[value].append(weight)
                else:
                    meta_data[value] = [freq]
                    meta_data_w[value] = [weight]

        for value in meta_data:
            weight_sum = sum(meta_data_w[value])
            if weight_sum < 1:
                meta_data[value].append(ECDF([0], [1], initialized=True))
                meta_data_w[value].append(1 - weight_sum)

        progress = meta_data.items()
        if progress_bar:
            progress = Progress(progress)
            progress.suffix = suffix
        meta_data = {
            value: ECDF.reduce(ecdfs, weights=meta_data_w[value], **kwargs)
            for value, ecdfs in progress
        }

        return super().reduce(structs, weights=weights, meta_data=meta_data, **kwargs)
