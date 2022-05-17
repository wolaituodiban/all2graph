from typing import Dict

import numpy as np
import pandas as pd

from .distribution import Distribution
from .ecdf import ECDF


class Discrete(Distribution):
    """离散分布"""
    def __init__(self, prob: Dict[str, float], **kwargs):
        super().__init__(**kwargs)
        self.prob = prob

    def __eq__(self, other, **kwargs):
        if len(self.prob) == len(other.prob) == len(set(self.prob).union(other.prob)):
            for k in self.prob:
                if not np.isclose(self.prob[k], other.prob[k]):
                    return False
            return True
        else:
            return False

    def __len__(self):
        return len(self.prob)

    def __getitem__(self, item):
        return self.prob[item]

    def __iter__(self):
        return iter(self.prob)

    def to_json(self) -> dict:
        output = super().to_json()
        output['prob'] = self.prob
        return output

    @classmethod
    def from_data(cls, array, **kwargs):
        value_counts = pd.value_counts(array, sort=False)
        num_samples = int(sum(value_counts))
        value_counts /= num_samples
        prob = value_counts.to_dict()
        return super().from_data(prob=prob, **kwargs)

    @classmethod
    def batch(cls, discretes, weights=None, **kwargs):
        discretes = list(discretes)
        if weights is None:
            weights = np.full(len(discretes), 1/len(discretes))
        else:
            weights = weights / sum(weights)

        prob = {}
        for discrete, w in zip(discretes, weights):
            for k, v in discrete.prob.items():
                if k in prob:
                    prob[k] += v * w
                else:
                    prob[k] = v * w
        return super().batch(discretes, prob=prob, **kwargs)

    @classmethod
    def from_ecdfs(cls, ecdfs: Dict[str, ECDF], weights: Dict[str, float] = None, **kwargs):
        if weights is None:
            weights = {k: 1/len(ecdfs) for k in ecdfs}
        prob = {k: v.mean * weights[k] for k, v in ecdfs.items()}
        value_sum = sum(prob.values())
        prob = {k: v / value_sum for k, v in prob.items()}
        return Discrete(prob=prob, initialized=True, **kwargs)
