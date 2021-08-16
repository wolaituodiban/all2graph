import numpy as np
import pandas as pd

from .distribution import Distribution
from ..macro import EPSILON


class ECDF(Distribution):
    """经验累计分布函数"""
    def __init__(self, x, y, **kwargs):
        """

        :param x: 随机变量的取值
        :param y: 取值对应的累积概率值
        :num_samples: 原始数据的数据量
        """
        super().__init__(**kwargs)
        self.x = np.array(x)
        self.y = np.array(y)

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
    def from_data(cls, array, **kwargs):
        # pd.value_counts sort by frequency，并不是我想要的功能
        counts = pd.value_counts(array, sort=False)
        counts = counts.sort_index(ascending=True)
        counts_cumsum = counts.cumsum()
        counts_cumsum /= counts_cumsum.iloc[-1]
        return super().from_data(x=counts_cumsum.index, y=counts_cumsum.values, **kwargs)

    @classmethod
    def reduce(cls, ecdfs, weights=None, **kwargs):
        """合并多个经验累计分布函数，返回一个贾总的经验累计分布函数"""
        # todo 优化reduce的速度以超过from_data的速度，否则就没有意义了
        if weights is None:
            weights = np.full(len(ecdfs), 1/len(ecdfs))
        else:
            weights = weights / sum(weights)
        temp = pd.concat([pd.Series(ecdf.y, index=ecdf.x) * w for ecdf, w in zip(ecdfs, weights)], axis=1)
        temp = temp.sort_index(ascending=True)
        temp.fillna(method='pad', inplace=True)
        temp = temp.sum(axis=1)
        return super().reduce(ecdfs, x=temp.index, y=temp.values, **kwargs)
