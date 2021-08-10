import json
from typing import Dict

import pandas as pd

from .string_node import StringNode
from ..macro import SECOND_DIFF
from ..stats import ECDF


ALL_TIME_UNITS = {'year', 'month', 'day', 'weekday', 'hour', 'minute', 'second'}
ALL_TIME_FEATURES = ALL_TIME_UNITS.union([SECOND_DIFF])


class TimeStamp(StringNode):
    """时间戳节点"""
    def __init__(self, value_dist: Dict[str, ECDF], **kwargs):
        """

        :param value_dist: 在node的角度上，每个时间特征的分布
        :param kwargs:
        """
        assert ALL_TIME_FEATURES.issuperset(value_dist), '衍生变量的范围必须在{}之内'.format(ALL_TIME_FEATURES)
        super().__init__(value_dist=value_dist, **kwargs)

    @classmethod
    def from_data(cls, num_samples, sample_ids, values, sample_time=None, **kwargs):
        value_dist = {}
        node_datetime = pd.to_datetime(values, utc=True)
        if sample_time is not None:
            sample_datetime = pd.to_datetime(sample_time, utc=True)
            # 精确到纳秒
            value_dist[SECOND_DIFF] = ECDF.from_data((sample_datetime - node_datetime) / pd.Timedelta(1))
        for time_unit in ALL_TIME_UNITS:
            num = ECDF.from_data(getattr(node_datetime, time_unit))
            if num.mean_var[1] > 0:
                value_dist[time_unit] = num
        kwargs[cls.VALUE_DIST] = value_dist
        return super(StringNode, cls).from_data(num_samples=num_samples, sample_ids=sample_ids, values=values, **kwargs)
