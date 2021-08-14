import time
import json
import os
import numpy as np
import pandas as pd
from toad.utils.progress import Progress
from all2graph.stats import ECDF


def test_one_sample():
    array = [2, 2]
    ecdf = ECDF.from_data(array)
    assert ecdf.mean_var == (2, 0), '{}'.format(ecdf.mean_var)


def test_not_eq():
    ecdf1 = ECDF.from_data([1, 2])
    ecdf2 = ECDF.from_data([2, 3])
    assert ecdf1 != ecdf2


def test_ecdf():
    arrays = []
    ecdfs = []
    for i in range(2, 100):
        if i % 50 == 0:
            array = np.random.random(i)
        else:
            array = np.random.randint(0, 10, i)
        ecdf = ECDF.from_data(array)
        assert len(ecdf.x.shape) == len(ecdf.y.shape) == 1, '必须是一维随机变量'
        assert ecdf.x.shape[0] == ecdf.y.shape[0] > 0, '随机变量的取值范围必须超过1个'
        assert ecdf.y[-1] == 1, '累计概率值的最后一个值必须是1, 但是得到{}'.format(ecdf.y[-1])
        assert np.min(ecdf.y) > 0, '累计概率值必须大于0'
        if ecdf.x.shape[0] > 1:
            assert np.min(np.diff(ecdf.x)) > 0, '随机变量的取值必须是单调的'
            assert np.min(np.diff(ecdf.y)) > 0, '累计概率值必须是单调的'
        assert np.abs(array.mean() - ecdf.mean) < 1e-5, 'test_mean failed, {} vs. {}'.format(array.mean(), ecdf.mean)
        mean, var = ecdf.mean_var
        assert np.abs(array.mean() - mean) < 1e-5, 'test_mean_var failed, {} vs. {}'.format(array.mean(), mean)
        assert np.abs(array.var() - var) < 1e-5, 'test_var failed, {} vs. {}'.format(array.std(), var)
        ecdf2 = ECDF.from_json(json.dumps(ecdf.to_json()))
        assert ecdf == ecdf2, '{} vs. {}'.format(ecdf.to_json(), ecdf2.to_json())
        arrays.append(array)
        ecdfs.append(ecdf)

    array = np.concatenate(arrays)
    ecdf = ECDF.reduce(ecdfs)
    mean, var = ecdf.mean_var
    assert np.abs(array.mean() - mean) < 1e-5, 'test_mean_var failed, {} vs. {}'.format(array.mean(), mean)
    assert np.abs(array.var() - var) < 1e-5, 'test_var failed, {} vs. {}'.format(array.std(), var)
    assert ecdf.num_samples == 4949
    assert ecdf.num_steps == 60


def speed():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.join(path, 'test_data', 'MensShoePrices', 'archive', 'train.csv')
    df = pd.read_csv(path)
    for col in df:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(axis=1, how='all')
    df = pd.concat([df])

    start_time = time.time()
    ecdfs = [
        ECDF.from_data(series) for col, series in df.iteritems()
    ]
    use_time = time.time() - start_time

    start_time = time.time()
    ecdf = ECDF.reduce(ecdfs)
    use_time2 = time.time() - start_time
    print(use_time, use_time2, ecdf.num_steps, ecdf.num_samples)
    print(sum(x.num_steps for x in ecdfs))
    assert use_time2 < use_time


if __name__ == '__main__':
    test_one_sample()
    test_not_eq()
    test_ecdf()
    speed()
    print('test_ecdf success')
