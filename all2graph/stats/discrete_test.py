import numpy as np
from all2graph.stats import Discrete


def test_descrete():
    array = ['a', 'a', 'b', 'c', None, np.nan]
    discrete = Discrete.from_data(array)
    assert abs(sum(discrete.frequency.values()) - 1) < 1e-5, '概率之和不为1'
    assert abs(discrete['a'] - 1/3) < 1e-5
    assert abs(discrete['b'] - 1/6) < 1e-5
    assert abs(discrete['c'] - 1/6) < 1e-5
    assert abs(discrete[None] - 1/3) < 1e-5

    json_obj = discrete.to_json()
    discrete = Discrete.from_json(json_obj)
    assert json_obj == discrete.to_json()
    print(json_obj)


if __name__ == '__main__':
    test_descrete()
    print('测试离散分布成功')
