import pandas as pd

from .meta_node import MetaNode
from ..stats import ECDF


class Number(ECDF, MetaNode):
    TOTAL_NUM_SAMPLES = 'total_num_samples'
    """数值型节点"""
    def __init__(self, x, y, num_samples, total_num_samples, **kwargs):
        """

        :param x: 随机变量的取值
        :param y: 取值对应的累积概率值
        :num_samples: 非空原始数据的数据量
        :param total_num_samples: 总原始数据的数据量
        """
        super().__init__(x=x, y=y, num_samples=num_samples, **kwargs)
        self.total_num_samples = total_num_samples

    def __eq__(self, other):
        return super().__eq__(other) and self.total_num_samples == other.total_num_samples

    @property
    def miss_rate(self):
        return 1 - self.num_samples / self.total_num_samples

    def to_json(self) -> dict:
        """返回可以被序列化的json对象"""
        output = super().to_json()
        output[self.TOTAL_NUM_SAMPLES] = self.total_num_samples
        return output

    @classmethod
    def from_data(cls, data, max_error_rate=None):
        """
        从序列中构造数值型节点
        :param data: 序列
        :param max_error_rate: 最大允许的不能转换成数值的比例
        :return:
        """
        array = pd.to_numeric(data, errors='coerce')

        if max_error_rate is not None:
            miss_rate = pd.isna(data).mean()
            assert array.isna().mean() - miss_rate <= max_error_rate, '无法转换的数据比例超过{}'.format(max_error_rate)

        ecdf = ECDF.from_data(array[pd.notna(array)])
        return cls(x=ecdf.x, y=ecdf.y, num_samples=ecdf.num_samples, total_num_samples=array.shape[0])

    @classmethod
    def merge(cls, ecdfs):
        raise NotImplementedError
