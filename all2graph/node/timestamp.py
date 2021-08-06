from typing import Dict

import pandas as pd

from .meta_node import MetaNode
from ..macro import SECOND_DIFF
from ..stats import ECDF


ALL_TIME_UNITS = {'year', 'month', 'day', 'weekday', 'hour', 'minute', 'second'}
ALL_TIME_FEATURES = ALL_TIME_UNITS.union([SECOND_DIFF])


class TimeStamp(MetaNode):
    """时间戳节点"""
    def __init__(self, ecdfs: Dict[str, ECDF], **kwargs):
        """

        :params ecdfs: 时间戳衍生变量的经验累计分布函数
        :params kwargs: MetaNode的参数
        """
        super().__init__(**kwargs)
        assert ALL_TIME_FEATURES.issubset(ecdfs), '衍生变量的范围必须在{}之内'.format(ALL_TIME_FEATURES)
        self.ecdfs = ecdfs

    def __eq__(self, other):
        return super().__eq__(other) and self.ecdfs == self.ecdfs


