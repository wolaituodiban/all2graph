import numpy as np
import pandas as pd
from scipy import interpolate

from .distribution import Distribution


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
            return (np.array(q) >= self.quantiles[0]).astype(float)
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
            return np.full_like(p, self.quantiles[0])
        else:
            return interpolate.interp1d(
                self.probs, self.quantiles, bounds_error=bounds_error,
                assume_sorted=assume_sorted, fill_value=fill_value, **kwargs
            )(p)

    def minmax_scale(self, x, prob_range=None, clip=False):
        """
        Args:
            x: 输入
            prob_range: 上下限的概率，如果对应的上下限是nan，那么会被替换成min或者max
            clip: 是否clip到0和1之间

        Returns:

        """
        if prob_range is None:
            lower, upper = self.minmax
        else:
            lower, upper = self.get_quantiles(prob_range)
            if np.isnan(lower):
                lower = self.min
            if np.isnan(upper):
                upper = self.max
        output = (x - lower) / (upper - lower)
        if clip:
            output = np.clip(output, 0, 1)
        return output

    def __eq__(self, other, **kwargs) -> bool:
        if self.quantiles.shape[0] == other.quantiles.shape[0] and self.probs.shape[0] == other.probs.shape[0]:
            return np.allclose(self.quantiles, other.quantiles) and np.allclose(self.probs, other.probs)
        else:

            return False

    @property
    def notna_quantiles(self):
        return self.quantiles[np.bitwise_not(np.isnan(self.quantiles))]

    @property
    def num_bins(self):
        return self.quantiles.shape[0]

    @property
    def max(self):
        return self.notna_quantiles[-1]

    @property
    def min(self):
        return self.notna_quantiles[-1]

    @property
    def minmax(self):
        return self.notna_quantiles[[0, -1]]

    @property
    def mean(self):
        return np.dot(self.quantiles, np.diff(self.probs, prepend=0))

    @property
    def mean_var(self):
        delta_prob = np.diff(self.probs, prepend=0)
        mean = np.dot(self.quantiles, delta_prob)
        square = (self.quantiles - mean) ** 2
        mask = np.bitwise_not(np.isinf(square))
        return mean, np.dot(square[mask], delta_prob[mask])

    def compress(self, num_bins):
        """
        压缩分箱的数量
        :param num_bins:
        :return:
        """
        # 一种能想到的指标是让min(diff(probs))最大化
        if self.num_bins <= num_bins:
            return
        probs = np.arange(0, 1+1/num_bins, 1/num_bins)
        probs = np.reshape(probs, (-1, 1))
        diff = np.abs(probs - self.probs)
        argmin = np.argmin(diff, axis=-1)
        self.probs = self.probs[argmin]
        self.quantiles = self.quantiles[argmin]

    def to_json(self) -> dict:
        output = super().to_json()
        output.update({
            'quantiles': self.quantiles.tolist(),
            'probs': self.probs.tolist()
        })
        return output

    @classmethod
    def from_data(cls, array, num_bins: int = None):
        # pd.value_counts sort by frequency，并不是我想要的功能
        counts = pd.value_counts(array, sort=False)
        counts = counts.sort_index(ascending=True)
        counts_cumsum = counts.cumsum()
        if counts_cumsum.shape[0] > 0:
            counts_cumsum /= counts_cumsum.iloc[-1]
        return super().from_data(quantiles=counts_cumsum.index, probs=counts_cumsum.values, num_bins=num_bins)

    @classmethod
    def batch(cls, structs, weights=None, num_bins=None):
        if weights is None:
            weights = np.full(len(structs), 1 / len(structs))
        else:
            weights = np.array(weights) / sum(weights)

        quantiles = [struct.quantiles for struct in structs]
        quantiles = np.concatenate(quantiles)
        quantiles = np.unique(quantiles)

        probs = [w * struct.get_probs(quantiles, kind='previous') for w, struct in zip(weights, structs)]
        probs = np.sum(probs, axis=0)
        return super().batch(structs, weights=weights, quantiles=quantiles, probs=probs, num_bins=num_bins)

    def extra_repr(self) -> str:
        if self.num_bins == 0:
            return str(np.nan)
        p = np.arange(0, 1, 0.25)[1:]
        q = self.get_quantiles(p)
        return ', '.join('{:.3}({:.3})'.format(x, y) for x, y in zip(q, p))

    def plot(self):
        import matplotlib.pyplot as plt
        return plt.plot(self.quantiles, self.probs)
