import numpy as np
import pandas as pd

from .distribution import Distribution
from ..macro import EPSILON


class ECDF(Distribution):
    """经验累计分布函数"""
    def __init__(self, x, y, num_samples: int, **kwargs):
        """

        :param x: 随机变量的取值
        :param y: 取值对应的累积概率值
        :num_samples: 原始数据的数据量
        """
        super().__init__(num_samples=num_samples, **kwargs)
        x = np.array(x)
        y = np.array(y)
        # assert len(x.shape) == len(y.shape) == 1, '必须是一维随机变量'
        # assert x.shape[0] == y.shape[0] > 0, '随机变量的取值范围必须超过1个'
        # assert y[-1] == 1, '累计概率值的最后一个值必须是1, 但是得到{}'.format(y[-1])
        # assert np.min(y) > 0, '累计概率值必须大于0'
        # if x.shape[0] > 1:
        #     assert np.min(np.diff(x)) > 0, '随机变量的取值必须是单调的'
        #     assert np.min(np.diff(y)) > 0, '累计概率值必须是单调的'
        self.x = x
        self.y = y

    def __eq__(self, other) -> bool:
        if super().__eq__(other) and self.x.shape[0] == other.x.shape[0] and self.y.shape[0] == other.y.shape[0]:
            return np.abs(self.x - other.x).max() <= EPSILON and np.abs(self.y - other.y).max() <= EPSILON
        else:
            return False

    @property
    def num_steps(self):
        return self.x.shape[0]

    @property
    def mean(self):
        return np.dot(self.x, np.diff(self.y, prepend=0))

    @property
    def mean_var(self):
        delta_prob = np.diff(self.y, prepend=0)
        mean = np.dot(self.x, delta_prob)
        return mean, np.dot((self.x - mean) ** 2, delta_prob)

    def to_json(self) -> dict:
        output = super().to_json()
        output.update({
            'x': self.x.tolist(),
            'y': self.y.tolist()
        })
        return output

    @classmethod
    def from_json(cls, obj):
        return super().from_json(obj)

    @classmethod
    def from_data(cls, array, **kwargs):
        # pd.value_counts sort by frequency，并不是我想要的功能
        counts = pd.value_counts(array, sort=False)
        counts = counts.sort_index(ascending=True)
        counts_cumsum = counts.cumsum()
        num_samples = int(counts_cumsum.iloc[-1])
        counts_cumsum /= num_samples
        return super().from_data(x=counts_cumsum.index, y=counts_cumsum.values, num_samples=num_samples, **kwargs)

    @classmethod
    def reduce(cls, ecdfs, **kwargs):
        """合并多个经验累计分布函数，返回一个贾总的经验累计分布函数"""
        # todo 判断reduce的速度是否小于from_data的速度，否则就没有意义了
        num_samples = 0
        counts = None
        for ecdf in ecdfs:
            new_value_counts = pd.Series(np.diff(ecdf.y, prepend=0) * ecdf.num_samples, index=ecdf.x)
            if counts is None:
                counts = new_value_counts
            else:
                counts = pd.concat([counts, new_value_counts], axis=1)
                counts = counts.sum(axis=1)
            num_samples += ecdf.num_samples
        counts = counts.sort_index(ascending=True)
        counts_cumsum = counts.cumsum()
        # 检测并修正最后一个计数为样本数
        assert abs(counts_cumsum.iloc[-1] - num_samples) < 1e-5, '{} v.s {}'.format(
            counts_cumsum.iloc[-1], num_samples
        )
        counts_cumsum.iloc[-1] = num_samples
        counts_cumsum /= num_samples
        return super().reduce(ecdfs, x=counts_cumsum.index, y=counts_cumsum.values, num_samples=num_samples, **kwargs)
