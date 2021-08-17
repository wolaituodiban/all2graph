from typing import Dict

import pandas as pd

from .meta_string import MetaString
from ....macro import SECOND_DIFF
from ....stats import ECDF


ALL_TIME_UNITS = {'year', 'month', 'day', 'weekday', 'hour', 'minute', 'second'}
ALL_TIME_FEATURES = ALL_TIME_UNITS.union([SECOND_DIFF])


class MetaTimeStamp(MetaString):
    """时间戳节点"""
    def __init__(self, meta_data: Dict[str, ECDF], **kwargs):
        """

        :param meta_data: 在node的角度上，每个时间特征的分布
        :param kwargs:
        """
        assert ALL_TIME_FEATURES.issuperset(meta_data), '衍生变量的范围必须在{}之内'.format(ALL_TIME_FEATURES)
        super().__init__(meta_data=meta_data, **kwargs)

    @classmethod
    def from_data(cls, num_samples, sample_ids, values, sample_times=None, **kwargs):
        if isinstance(values, pd.Series):
            values = values.values
        meta_data = {}
        node_datetime = pd.to_datetime(values, utc=True, errors='coerce')
        if sample_times is not None:
            sample_datetime = pd.to_datetime(sample_times, utc=True, errors='coerce')
            # 精确到纳秒
            meta_data[SECOND_DIFF] = ECDF.from_data((sample_datetime-node_datetime)/pd.Timedelta(1)/1e9, **kwargs)
        for time_unit in ALL_TIME_UNITS:
            num = ECDF.from_data(getattr(node_datetime, time_unit), **kwargs)
            if num.mean_var[1] > 0:
                meta_data[time_unit] = num
        return super(MetaString, cls).from_data(
            num_samples=num_samples, sample_ids=sample_ids, values=values, meta_data=meta_data, **kwargs
        )
