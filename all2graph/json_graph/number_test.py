import json
import numpy as np
import pandas as pd
from all2graph.json_graph import Number


def test_number():
    for i in range(2, 100):
        if i % 50 == 0:
            array = np.random.random(i)
        else:
            array = np.random.randint(0, 10, i)

        array = pd.Series(array)
        array.iloc[np.random.binomial(2, 0.2, i).astype(bool)] = 'haha'

        if (array == 'haha').mean() > 0.2:
            try:
                Number.from_data(array, 0.2)
                raise AssertionError('最大允许数值转换错误比例测试失败')
            except AssertionError:
                pass
        elif pd.to_numeric(array, errors='coerce').notna().sum() > 1:
            num = Number.from_data(array, 0.2)
            assert np.abs(num.miss_rate - (array == 'haha').mean()) < 1e-5, '{} vs. {}'.format(
                num.miss_rate, (array == 'haha').mean()
            )

            array = pd.to_numeric(array, errors='coerce')
            assert np.abs(array.mean() - num.mean) < 1e-5, 'test_mean failed, {} vs. {}'.format(array.mean(), num.mean)

            mean, var = num.mean_var
            assert np.abs(array.mean() - mean) < 1e-5, 'test_mean_var failed, {} vs. {}'.format(array.mean(), mean)
            assert np.abs(np.var(array) - var) < 1e-5, '{}, {}, {}, {}, {}'.format(
                np.var(array), var, array, num.ecdf.x, num.ecdf.y
            )

            json_obj = num.to_json()
            num = Number.from_json(json_obj)
            assert num.to_json() == json_obj, '{} vs. {}'.format(num.to_json(), json_obj)
    print(json.dumps(num.to_json(), indent=2))
    print('test_number success')


if __name__ == '__main__':
    test_number()
    print('测试Number成功')
