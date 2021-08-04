import numpy as np
import pandas as pd
from typing import Dict
from .distribution import Distribution


class Discrete(Distribution):
    """离散分布"""
    def __init__(self, frequency: Dict[str, float], num_samples, **kwargs):
        super().__init__(num_samples=num_samples, **kwargs)
        self.frequency = frequency

    def __getitem__(self, item):
        return self.frequency[item]

    def to_json(self) -> dict:
        output = super().to_json()
        output['frequency'] = self.frequency
        return output

    @classmethod
    def from_data(cls, array):
        value_counts = pd.value_counts(array, dropna=False)
        num_samples = value_counts.sum()
        value_counts /= num_samples
        frequency = value_counts.to_dict()
        if np.nan in frequency:
            frequency[None] = frequency[np.nan]
            del frequency[np.nan]

        return cls(frequency=frequency, num_samples=num_samples)
