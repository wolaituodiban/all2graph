import json
import numpy as np
from all2graph.stats import ECDF


def test_one_sample():
    array = [2, 2]
    ecdf = ECDF.from_data(array)
    assert ecdf.mean_var == (2, 0)


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
        assert np.abs(array.mean() - ecdf.mean) < 1e-5, 'test_mean failed, {} vs. {}'.format(array.mean(), ecdf.mean)
        mean, var = ecdf.mean_var
        assert np.abs(array.mean() - mean) < 1e-5, 'test_mean_var failed, {} vs. {}'.format(array.mean(), mean)
        assert np.abs(array.var() - var) < 1e-5, 'test_var failed, {} vs. {}'.format(array.std(), var)
        ecdf2 = ECDF.from_json(json.dumps(ecdf.to_json()))
        assert ecdf == ecdf2, '{} vs. {}'.format(ecdf.to_json(), ecdf2.to_json())
        arrays.append(array)
        ecdfs.append(ecdf)

    array = np.concatenate(arrays)
    ecdf = ECDF.merge(ecdfs)
    mean, var = ecdf.mean_var
    assert np.abs(array.mean() - mean) < 1e-5, 'test_mean_var failed, {} vs. {}'.format(array.mean(), mean)
    assert np.abs(array.var() - var) < 1e-5, 'test_var failed, {} vs. {}'.format(array.std(), var)
    assert ecdf.num_samples == 4949
    assert ecdf.num_steps == 60
    print(json.dumps(ecdf.to_json(), indent=2))


if __name__ == '__main__':
    test_one_sample()
    test_not_eq()
    test_ecdf()
    print('test_ecdf success')
