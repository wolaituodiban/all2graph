import json
from typing import Dict

import pandas as pd

from .category import Category
from ..macro import SECOND_DIFF
from ..stats import ECDF


ALL_TIME_UNITS = {'year', 'month', 'day', 'weekday', 'hour', 'minute', 'second'}
ALL_TIME_FEATURES = ALL_TIME_UNITS.union([SECOND_DIFF])


class TimeStamp(Category):
    """时间戳节点"""
    def __init__(self, ecdfs: Dict[str, ECDF], **kwargs):
        """

        :param ecdfs: 在node的角度上，每个时间特征的分布
        :param kwargs:
        """
        assert ALL_TIME_FEATURES.issuperset(ecdfs), '衍生变量的范围必须在{}之内'.format(ALL_TIME_FEATURES)
        super().__init__(ecdfs=ecdfs, **kwargs)

    @property
    def num_nodes(self) -> int:
        return self[list(self.ecdfs)[0]].num_samples

    @classmethod
    def from_data(cls, node_time, sample_time=None, **kwargs):
        ecdfs = {}
        node_datetime = pd.to_datetime(node_time, utc=True)
        if sample_time is not None:
            sample_datetime = pd.to_datetime(sample_time, utc=True)
            # 精确到纳秒
            ecdfs[SECOND_DIFF] = ECDF.from_data((sample_datetime - node_datetime) / pd.Timedelta(1))
        for time_unit in ALL_TIME_UNITS:
            num = ECDF.from_data(getattr(node_datetime, time_unit))
            if num.mean_var[1] > 0:
                ecdfs[time_unit] = num
        return super(Category, cls).from_data(node_time, ecdfs=ecdfs, **kwargs)
