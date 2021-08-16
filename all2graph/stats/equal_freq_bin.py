import numpy as np
import pandas as pd

from .distribution import Distribution
from ..macro import EPSILON


class EqualFreqBin(Distribution):
    def __init__(self, quantiles, bins, **kwargs):
        super().__init__(**kwargs)
        self.quantiles = np.array(quantiles)
        self.bins = np.array(bins)

    def __eq__(self, other):
        return super().__eq__(other) and max(abs(self.bins - self.bins)) < EPSILON

    @property
    def _freqs(self):
        return np.diff(self.quantiles)

    @staticmethod
    def _mean(freqs, midpoints):
        return np.dot(freqs, midpoints)

    @property
    def _midpoints(self):
        midpoints = (self.bins[1:] + self.bins[:-1]) / 2
        return midpoints

    @property
    def mean(self):
        return self._mean(self._freqs, self._midpoints)

    @property
    def mean_var(self):
        midpoints = self._midpoints
        freqs = self._freqs
        mean = self._mean(freqs, midpoints)
        var = np.dot(freqs, (midpoints - mean) ** 2)
        return mean, var

    def to_json(self):
        output = super().to_json()
        output['quantiles'] = self.quantiles.tolist()
        output['bins'] = self.bins.tolist()
        return output

    @classmethod
    def from_data(cls, numbers, num_bins, **kwargs):
        out, bins = pd.qcut(numbers, num_bins, retbins=True, duplicates='drop')
        counts = pd.value_counts(out, sort=False).values
        quantiles = counts / counts[-1]
        return super().from_data(quantiles=quantiles, bins=bins, **kwargs)

    @classmethod
    def reduce(cls, structs, num_bins, weights=None, **kwargs):

        if weights is None:
            weights = np.full(len(structs), 1 / len(structs))
        else:
            weights = weights / sum(weights)


