import json
from typing import Dict

import pandas as pd

from .meta_node import MetaNode
from .number import Number
from ..macro import SECOND_DIFF


ALL_TIME_UNITS = {'year', 'month', 'day', 'weekday', 'hour', 'minute', 'second'}
ALL_TIME_FEATURES = ALL_TIME_UNITS.union([SECOND_DIFF])


class TimeStamp(MetaNode):
    FEATS = 'feats'
    """时间戳节点"""
    def __init__(self, feats: Dict[str, Number], **kwargs):
        """

        :params feats: 时间戳衍生变量的经验累计分布函数
        :params kwargs: MetaNode的参数
        """
        super().__init__(**kwargs)
        assert ALL_TIME_FEATURES.issuperset(feats), '衍生变量的范围必须在{}之内'.format(ALL_TIME_FEATURES)
        assert all(isinstance(feat, Number) for feat in feats.values()), 'feats必须是Number'
        assert len(set(feat.total_num_samples for feat in feats.values())) == 1, '所有feats的total_num_samples必须相同'
        self.feats = feats

    @property
    def total_num_samples(self):
        return self.feats[SECOND_DIFF].total_num_samples

    def __eq__(self, other):
        return super().__eq__(other) and self.feats == other.feats

    def to_json(self) -> dict:
        output = super().to_json()
        output[self.FEATS] = {k: v.to_json() for k, v in self.feats.items()}
        return output

    @classmethod
    def from_json(cls, obj):
        if isinstance(obj, str):
            obj = json.loads(obj)
        else:
            obj = dict(obj)
        obj[cls.FEATS] = {k: Number.from_json(v) for k, v in obj[cls.FEATS].items()}
        return super().from_json(obj)

    @classmethod
    def from_data(cls, node_time, sample_time=None, max_error_rate=None, **kwargs):
        node_datetime = pd.to_datetime(node_time, errors='coerce', utc=True)
        if max_error_rate is not None:
            assert node_datetime.isna().mean() - pd.isna(node_time).mean() <= max_error_rate, '无法转换的数据比例超过'
        sample_datetime = pd.to_datetime(sample_time, utc=True)
        feats = {
            # 精确到纳秒
            SECOND_DIFF: Number.from_data((sample_datetime - node_datetime) / pd.Timedelta(1))
        }
        for time_unit in ALL_TIME_UNITS:
            num = Number.from_data(getattr(node_datetime, time_unit))
            if num.mean_var[1] > 0:
                feats[time_unit] = num
        return super().from_data(node_time, feats=feats, **kwargs)

    @classmethod
    def merge(cls, timestamps, **kwargs):
        feats = {}
        total_num_samples = 0
        for timestamp in timestamps:
            total_num_samples += timestamp.total_num_samples
            for value, feat in timestamp.feats.items():
                if value not in feats:
                    feats[value] = [feat]
                else:
                    feats[value].append(feat)
        feats = {k: Number.merge(v) for k, v in feats.items()}
        # 将所有值的频率分布补0，直到样本数一致
        for value, feat in feats.items():
            if feat.total_num_samples < total_num_samples:
                feat.total_num_samples = total_num_samples
        return super().from_data(timestamps, feats=feats, **kwargs)
