import pandas as pd
from typing import Dict
from .distribution import Distribution


class Discrete(Distribution):
    """离散分布"""
    def __init__(self, prob: Dict[str, float], num_samples, **kwargs):
        super().__init__(num_samples=num_samples, **kwargs)
        assert abs(sum(prob.values()) - 1) < 1e-5, '概率之和{}不为1'.format(sum(prob.values()))
        self.prob = prob

    def to_json(self) -> dict:
        output = super().to_json()
        output['prob'] = self.prob
        return output

    @classmethod
    def from_data(cls, array):
        value_counts = pd.value_counts(array)
        num_samples = len(array)
        value_counts[None] = num_samples - value_counts.sum()
        value_counts /= num_samples
        prob = value_counts.to_dict()
        return cls(prob=prob, num_samples=num_samples)

    @classmethod
    def merge(cls, discretes):
        value_counts = {}
        num_samples = 0
        for discrete in discretes:
            num_samples += discrete.num_samples
            for k, v in discrete.prob.items():
                if k in value_counts:
                    value_counts[k] += v * discrete.num_samples
                else:
                    value_counts[k] = v * discrete.num_samples
        prob = {k: v / num_samples for k, v in value_counts.items()}
        return cls(prob=prob, num_samples=num_samples)
