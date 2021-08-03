import numpy as np
import pandas as pd

from ..meta_struct import MetaStruct


class ECDF(MetaStruct):
    """经验累计分布函数"""
    def __init__(self, x, y, num_samples: int):
        """

        :param x: 随机变量的取值
        :param y: 取值对应的累积概率值
        :num_samples: 原始数据的数据量
        """
        super().__init__()
        x = np.array(x)
        y = np.array(y)
        assert len(x.shape) == len(y.shape) == 1, '必须是一维随机变量'
        assert x.shape[0] == y.shape[0] > 1, '随机变量的取值范围必须超过1个'
        assert y[-1] == 1, '累计概率值的最后一个值必须是1, 但是得到{}'.format(y[-1])
        assert np.min(y) > 0, '累计概率值必须大于0'
        assert np.min(np.diff(x)) > 0, '随机变量的取值必须是单调的'
        assert np.min(np.diff(y)) > 0, '累计概率值必须是单调的'
        self.x = x
        self.y = y
        self.num_samples = num_samples

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

    def to_json(self):
        output = super().to_json()
        output.update({
            'x': self.x.tolist(),
            'y': self.y.tolist(),
            'num_samples': self.num_samples
        })
        return output

    @classmethod
    def from_json(cls, obj):
        return super().from_json(obj)

    @classmethod
    def from_array(cls, array):
        array = pd.Series(array)
        value_counts = array.value_counts()
        value_counts = value_counts.reset_index()
        value_counts = value_counts.sort_values('index')
        value_counts['y'] = value_counts[0].cumsum()
        assert array.shape[0] == value_counts['y'].iloc[-1]
        value_counts['y'] /= array.shape[0]
        return cls(value_counts['index'].values,  value_counts['y'].values, array.shape[0])

    @classmethod
    def merge(cls, ecdfs):
        """合并多个经验累计分布函数，返回一个贾总的经验累计分布函数"""
        num_samples = ecdfs[0].num_samples
        value_counts = pd.Series(np.diff(ecdfs[0].y, prepend=0) * ecdfs[0].num_samples, index=ecdfs[0].x)
        for ecdf in ecdfs[1:]:
            new_value_counts = pd.Series(np.diff(ecdf.y, prepend=0) * ecdf.num_samples, index=ecdf.x)
            value_counts = pd.concat([value_counts, new_value_counts], axis=1)
            value_counts = value_counts.sum(axis=1)
            num_samples += ecdf.num_samples
        value_counts = value_counts.reset_index()
        value_counts = value_counts.sort_values('index')
        value_counts['y'] = value_counts[0].cumsum()
        assert value_counts['y'].iloc[-1] == num_samples
        value_counts['y'] /= num_samples
        return cls(value_counts['index'].values,  value_counts['y'].values, num_samples)
