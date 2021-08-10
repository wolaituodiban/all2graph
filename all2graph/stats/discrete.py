import json
import pandas as pd
from typing import Dict
from .distribution import Distribution
from ..macro import EPSILON


class Discrete(Distribution):
    """离散分布"""
    def __init__(self, prob: Dict[str, float], num_samples, **kwargs):
        super().__init__(num_samples=num_samples, **kwargs)
        assert abs(sum(prob.values()) - 1) < EPSILON, '概率之和{}不为1'.format(sum(prob.values()))
        self.prob = prob

    def __eq__(self, other):
        if super().__eq__(other) and len(self.prob) == len(other.prob) == len(set(self.prob).union(other.prob)):
            for k in self.prob:
                if abs(self.prob[k] - other.prob[k]) > EPSILON:
                    return False
            return True
        else:
            return False

    def to_json(self) -> dict:
        output = super().to_json()
        output['prob'] = self.prob
        return output

    @classmethod
    def from_json(cls, obj):
        if isinstance(obj, str):
            obj = json.loads(obj)
        else:
            obj = dict(obj)
        if 'null' in obj['prob']:
            assert None not in obj['prob']
            obj['prob'][None] = obj['prob']['null']
            del obj['prob']['null']
        return super().from_json(obj)

    @classmethod
    def from_data(cls, array, **kwargs):
        value_counts = pd.value_counts(array)
        num_samples = len(array)
        value_counts[None] = num_samples - value_counts.sum()
        value_counts /= num_samples
        prob = value_counts.to_dict()
        return super().from_data(prob=prob, num_samples=num_samples, **kwargs)

    @classmethod
    def merge(cls, discretes, **kwargs):
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
        return super().merge(discretes, prob=prob, num_samples=num_samples, **kwargs)
