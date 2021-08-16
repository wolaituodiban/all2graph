from typing import Iterable

import numpy as np

from .distribution import Distribution
from ..macro import EPSILON


class EqualFreqBin(Distribution):
    def __init__(self, split_points, **kwargs):
        super().__init__(**kwargs)
        self.split_points = np.array(split_points)

    def __eq__(self, other):
        return super().__eq__(other) and max(abs(self.split_points - self.split_points)) < EPSILON

    @property
    def mean(self):
        return np.mean(self.split_points)

    @property
    def std(self):
        return np.std(self.split_points)

    def to_json(self):
        output = super().to_json()
        output['split_points'] = self.split_points.tolist()
        return output

    @classmethod
    def from_data(cls, numbers, bins, **kwargs):
        return super().from_data(
            split_points=np.quantile(numbers, np.arange(0, 1, 1/bins)[1:]),
            **kwargs
        )

    @classmethod
    def reduce(cls, structs, weights=None, **kwargs):
        if weights is None:
            weights = np.full(len(structs), 1 / len(structs))
        else:
            weights = weights / sum(weights)


