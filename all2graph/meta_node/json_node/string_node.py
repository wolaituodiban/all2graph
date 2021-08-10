import json
from typing import Dict

import pandas as pd

from all2graph.meta_node.meta_node import MetaNode
from all2graph.stats import Discrete, ECDF


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

    def __eq__(self, other):
        return super().__eq__(other) and self.value_dist == other.value_dist

    def to_discrete(self) -> Discrete:
        # todo
        pass

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
        value_dists = {}
        id_col = 'id'
        value_col = 'value'
        df = pd.DataFrame({id_col: sample_ids, value_col: values})
        count_df = df.reset_index().groupby([id_col, value_col]).count()
        num_nodes = count_df.index.get_level_values(0).unique().shape[0]
        for value, count in count_df.groupby(level=value_col):
            freq = count.values[:, 0].tolist()
            if len(freq) < num_nodes:
                freq += [0] * (num_nodes - len(freq))
            value_dists[value] = ECDF.from_data(freq)
        kwargs[cls.VALUE_DIST] = value_dists
        return super().from_data(num_samples=num_samples, sample_ids=sample_ids, values=values, **kwargs)

    @classmethod
    def reduce(cls, cats, **kwargs):
        valud_dists = {}
        num_samples = 0
        for cat in cats:
            num_samples += cat[list(cat.value_dist)[0]].num_samples
            for value, freq in cat.value_dist.items():
                if value not in valud_dists:
                    valud_dists[value] = [freq]
                else:
                    valud_dists[value].append(freq)
        valud_dists = {k: ECDF.reduce(v) for k, v in valud_dists.items()}
        # 将所有值的频率分布补0，直到样本数一致
        for value, freq in valud_dists.items():
            if freq.num_samples < num_samples:
                valud_dists[value] = ECDF.reduce([freq, ECDF.from_data([0] * (num_samples - freq.num_samples), **kwargs)])
        kwargs[cls.VALUE_DIST] = valud_dists
        return super().reduce(cats, **kwargs)
