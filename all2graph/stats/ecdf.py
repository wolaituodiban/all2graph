import numpy as np
import pandas as pd
from scipy import interpolate

from .distribution import Distribution
from ..macro import EPSILON


class ECDF(Distribution):
    """经验累计分布函数"""
    def __init__(self, quantiles, probs, num_bins=None, **kwargs):
        """

        :param quantiles: 分位数
        :param probs: 分位数的累积概率值
        :num_samples: 原始数据的数据量
        """
        super().__init__(**kwargs)
        self.quantiles = np.array(quantiles)
        self.probs = np.array(probs)

        if num_bins is not None:
            self.compress(num_bins)

    def get_probs(
            self, q, bounds_error=False, fill_value=(0, 1), assume_sorted=True, **kwargs
    ) -> np.ndarray:
        """

        :param q: 分位数
        :param bounds_error:
        :param fill_value:
        :param assume_sorted:
        :param kwargs:
        :return: 分位数对应的累计概率
        """
        if self.num_bins == 1:
            return np.array(q) >= self.quantiles[0]
        else:
            return interpolate.interp1d(
                self.quantiles, self.probs, bounds_error=bounds_error, fill_value=fill_value,
                assume_sorted=assume_sorted, **kwargs
            )(q)

    def get_quantiles(self, p, bounds_error=False, fill_value="extrapolate", assume_sorted=True,
                      **kwargs) -> np.ndarray:
        """

        :param p: 累积概率
        :param bounds_error:
        :param assume_sorted:
        :param fill_value:
        :param kwargs:
        :return: 累积概率对应的分位数
        """
        if self.num_bins == 1:
            return np.full_like(p, self.probs[0])
        else:
            return interpolate.interp1d(
                self.probs, self.quantiles, bounds_error=bounds_error,
                assume_sorted=assume_sorted, fill_value=fill_value, **kwargs
            )(p)

    def __eq__(self, other) -> bool:
        if super().__eq__(other) \
                and self.quantiles.shape[0] == other.quantiles.shape[0] \
                and self.probs.shape[0] == other.probs.shape[0]:
            return np.abs(self.quantiles - other.quantiles).max() <= EPSILON \
                   and np.abs(self.probs - other.probs).max() <= EPSILON
        else:
            return False

    @property
    def num_bins(self):
        return self.quantiles.shape[0]

    @property
    def mean(self):
        return np.dot(self.quantiles, np.diff(self.probs, prepend=0))

    @property
    def mean_var(self):
        delta_prob = np.diff(self.probs, prepend=0)
        mean = np.dot(self.quantiles, delta_prob)
        return mean, np.dot((self.quantiles - mean) ** 2, delta_prob)

    def compress(self, num_bins):
        """
        压缩分箱的数量
        :param num_bins:
        :return:
        """
        # todo 为了保证信息最多，压缩后的点的概率应该尽可能均匀的分布在[0, 1]之间
        # 一种能想到的指标是让min(diff(probs))最大化
        if self.num_bins <= num_bins:
            return
        delta_probs = np.diff(self.probs, prepend=0)
        delta_quantiles = np.diff(self.quantiles, prepend=self.quantiles[0]-1)
        fst_ord_diff = delta_probs / delta_quantiles
        fst_ord_diff[[0, -1]] = np.inf
        sel_mask = np.argsort(fst_ord_diff) >= self.num_bins - num_bins
        self.probs = self.probs[sel_mask]
        self.quantiles = self.quantiles[sel_mask]

    def to_json(self) -> dict:
        output = super().to_json()
        output.update({
            'quantiles': self.quantiles.tolist(),
            'probs': self.probs.tolist()
        })
        return output

    @classmethod
    def from_data(cls, array, num_bins: int = None, **kwargs):
        # pd.value_counts sort by frequency，并不是我想要的功能
        counts = pd.value_counts(array, sort=False)
        counts = counts.sort_index(ascending=True)
        counts_cumsum = counts.cumsum()
        counts_cumsum /= counts_cumsum.iloc[-1]
        return super().from_data(quantiles=counts_cumsum.index, probs=counts_cumsum.values, num_bins=num_bins, **kwargs)

    @classmethod
    def reduce(cls, structs, weights=None, **kwargs):
        if weights is None:
            weights = np.full(len(structs), 1 / len(structs))
        else:
            weights = np.array(weights) / sum(weights)

        quantiles = [struct.quantiles for struct in structs]
        quantiles = np.concatenate(quantiles)
        quantiles = np.unique(quantiles)

        probs = [w * struct.get_probs(quantiles, kind='previous') for w, struct in zip(weights, structs)]
        probs = np.sum(probs, axis=0)

        return super().reduce(structs, weights=weights, quantiles=quantiles, probs=probs, **kwargs)
