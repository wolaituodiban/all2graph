import os
import time
import json
import pandas as pd
import numpy as np
from all2graph.stats import Discrete


def test_descrete():
    array = ['a', 'a', 'b', 'c', None, np.nan]
    discrete = Discrete.from_data(array)
    assert abs(sum(discrete.prob.values()) - 1) < 1e-5, '概率之和不为1'
    assert abs(discrete.prob['a'] - 1/2) < 1e-5
    assert abs(discrete.prob['b'] - 1/4) < 1e-5
    assert abs(discrete.prob['c'] - 1/4) < 1e-5

    discrete2 = Discrete.from_json(json.loads(json.dumps(discrete.to_json())))
    assert discrete == discrete2, '{}\n{}'.format(discrete.to_json(), discrete2.to_json())
    print(discrete.to_json())


def test_not_eq():
    dis1 = Discrete.from_data(['a', 'a', 'b'])
    dis2 = Discrete.from_data(['a', 'b', 'b'])
    assert dis1 != dis2


def test_merge():
    arrays = []
    discretes = []
    for i in range(1, 100):
        array = np.random.choice(['a', 'b', 'c', None], size=i, replace=True)
        discrete = Discrete.from_data(array)
        arrays.append(array)
        discretes.append(discrete)

    discrete1 = Discrete.batch(discretes, weights=np.array([pd.notna(a).sum() for a in arrays]))
    discrete2 = Discrete.from_data(np.concatenate(arrays))
    assert discrete1 == discrete2, '{}\n{}'.format(discrete1.to_json(), discrete2.to_json())
    print(discrete1.to_json())


if __name__ == '__main__':
    test_descrete()
    test_merge()
