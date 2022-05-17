import time
import json
import os
import numpy as np
import pandas as pd
from all2graph.stats import ECDF


def test_one_sample():
    array = [2, 2]
    ecdf = ECDF.from_data(array)
    print(ecdf)
    assert ecdf.mean_var == (2, 0), '{}'.format(ecdf.mean_var)


def test_not_eq():
    ecdf1 = ECDF.from_data([1, 2])
    ecdf2 = ECDF.from_data([2, 3])
    print(ecdf1)
    print(ecdf2)
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
        assert len(ecdf.quantiles.shape) == len(ecdf.probs.shape) == 1, '必须是一维随机变量'
        assert ecdf.quantiles.shape[0] == ecdf.probs.shape[0] > 0, '随机变量的取值范围必须超过1个'
        assert ecdf.probs[-1] == 1, '累计概率值的最后一个值必须是1, 但是得到{}'.format(ecdf.probs[-1])
        assert np.min(ecdf.probs) > 0, '累计概率值必须大于0'
        if ecdf.quantiles.shape[0] > 1:
            assert np.min(np.diff(ecdf.quantiles)) > 0, '随机变量的取值必须是单调的'
            assert np.min(np.diff(ecdf.probs)) > 0, '累计概率值必须是单调的'
        assert np.abs(array.mean() - ecdf.mean) < 1e-5, 'test_mean failed, {} vs. {}'.format(array.mean(), ecdf.mean)
        mean, var = ecdf.mean_var
        assert np.abs(array.mean() - mean) < 1e-5, 'test_mean_var failed, {} vs. {}'.format(array.mean(), mean)
        assert np.abs(array.var() - var) < 1e-5, 'test_var failed, {} vs. {}'.format(array.std(), var)
        ecdf2 = ECDF.from_json(json.loads(json.dumps(ecdf.to_json())))
        assert ecdf == ecdf2, '{} vs. {}'.format(ecdf.to_json(), ecdf2.to_json())
        arrays.append(array)
        ecdfs.append(ecdf)

    array = np.concatenate(arrays)
    ecdf = ECDF.batch(ecdfs, weights=np.array([a.shape[0] for a in arrays]))
    print(ecdf)
    mean, var = ecdf.mean_var
    assert np.abs(array.mean() - mean) < 1e-5, 'test_mean_var failed, {} vs. {}'.format(array.mean(), mean)
    assert np.abs(array.var() - var) < 1e-5, 'test_var failed, {} vs. {}'.format(array.std(), var)
    assert ecdf.num_bins == 60


def test_compress():
    def q_diff(i, s, f, **kwargs):
        from scipy.stats import ks_2samp
        fq = f.get_quantiles(probs, **kwargs)
        stats, pvalue = ks_2samp(s, fq)
        assert pvalue > 0.05, '{} {} {} {}\n{}\n{}\n{}\n{}'.format(
            i, f.num_bins, stats, pvalue, np.quantile(s, probs), fq, f.quantiles, f.probs
        )

    # !!!本测试用于检验压缩算法的精度
    num_loops = 100
    num_samples = 1000
    bins = [512, 256, 128, 64, 32, 16]
    probs = np.arange(0, 1, 0.01)[1:]
    # beta
    print('beta')
    for b in bins:
        for _ in range(num_loops):
            samples = np.random.beta(5, 1, num_samples)
            ecdf = ECDF.from_data(samples, num_bins=b)
            assert ecdf.quantiles[0] == samples.min() and ecdf.quantiles[-1] == samples.max()
            assert ecdf.probs.min() >= 0 and ecdf.probs.max() <= 1
            q_diff(_, samples, ecdf)

    # binomial
    print('binomial')
    for b in bins:
        for _ in range(num_loops):
            samples = np.random.binomial(100, 0.2, num_samples)
            ecdf = ECDF.from_data(samples, num_bins=b)
            assert ecdf.quantiles[0] == samples.min() and ecdf.quantiles[-1] == samples.max()
            assert ecdf.probs.min() >= 0 and ecdf.probs.max() <= 1
            q_diff(_, samples, ecdf)

    print('chisquare')
    for b in bins:
        for _ in range(num_loops):
            samples = np.random.chisquare(5, num_samples)
            ecdf = ECDF.from_data(samples, num_bins=b)
            assert ecdf.quantiles[0] == samples.min() and ecdf.quantiles[-1] == samples.max()
            assert ecdf.probs.min() >= 0 and ecdf.probs.max() <= 1
            q_diff(_, samples, ecdf)

    print('exponential')
    for b in bins:
        for _ in range(num_loops):
            samples = np.random.exponential(3.14, num_samples)
            ecdf = ECDF.from_data(samples, num_bins=b)
            assert ecdf.quantiles[0] == samples.min() and ecdf.quantiles[-1] == samples.max()
            assert ecdf.probs.min() >= 0 and ecdf.probs.max() <= 1
            q_diff(_, samples, ecdf)

    print('gamma')
    for b in bins:
        for _ in range(num_loops):
            samples = np.random.gamma(2, 2, size=num_samples)
            ecdf = ECDF.from_data(samples, num_bins=b)
            assert ecdf.quantiles[0] == samples.min() and ecdf.quantiles[-1] == samples.max()
            assert ecdf.probs.min() >= 0 and ecdf.probs.max() <= 1
            q_diff(_, samples, ecdf)

    print('lognormal')
    for b in bins:
        for _ in range(num_loops):
            samples = np.random.lognormal(2, 2, size=num_samples)
            ecdf = ECDF.from_data(samples, num_bins=b)
            assert ecdf.quantiles[0] == samples.min() and ecdf.quantiles[-1] == samples.max()
            assert ecdf.probs.min() >= 0 and ecdf.probs.max() <= 1
            q_diff(_, samples, ecdf)

    print('logistic')
    for b in bins:
        for _ in range(num_loops):
            samples = np.random.logistic(1, 0.9, size=num_samples)
            ecdf = ECDF.from_data(samples, num_bins=b)
            assert ecdf.quantiles[0] == samples.min() and ecdf.quantiles[-1] == samples.max()
            assert ecdf.probs.min() >= 0 and ecdf.probs.max() <= 1
            q_diff(_, samples, ecdf)

    # print('poisson')
    # for b in bins:
    #     for _ in range(num_loops):
    #         samples = np.random.poisson(1, size=num_samples)
    #         ecdf = ECDF.from_data(samples, num_bins=b)
    #         assert ecdf.quantiles[0] == samples.min() and ecdf.quantiles[-1] == samples.max()
    #         assert ecdf.probs.min() >= 0 and ecdf.probs.max() <= 1
    #         q_diff(_, samples, ecdf, kind='previous', fill_value=(0, np.nan))

    print('wald')
    for b in bins:
        for _ in range(num_loops):
            samples = np.random.wald(1, 8, size=num_samples)
            ecdf = ECDF.from_data(samples, num_bins=b)
            assert ecdf.quantiles[0] == samples.min() and ecdf.quantiles[-1] == samples.max()
            assert ecdf.probs.min() >= 0 and ecdf.probs.max() <= 1
            q_diff(_, samples, ecdf)


def test_minmax_scale():
    a = [0, 1, 2, 3, 3, 4, 2, 1]
    ecdf = ECDF.from_data(a, num_bins=10)
    b = ecdf.minmax_scale(a)
    assert np.isclose(b, [0, 0.25, 0.5, 0.75, 0.75, 1, 0.5, 0.25]).all(), b
    c = [1, 5]
    d = ecdf.minmax_scale(c, prob_range=(0.5, 1))
    assert np.isclose(d, [-0.2, 1.4]).all(), d
    e = ecdf.minmax_scale(c, clip=True)
    assert np.isclose(e, [0.25, 1]).all(), e
    f = ecdf.minmax_scale(c, prob_range=(0.5, 0.9), clip=True)
    assert np.isclose(f, [0, 1]).all(), f


if __name__ == '__main__':
    test_one_sample()
    test_not_eq()
    test_ecdf()
    test_compress()
    test_minmax_scale()
