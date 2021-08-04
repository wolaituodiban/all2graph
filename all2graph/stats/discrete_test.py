import numpy as np
from all2graph.stats import Discrete
from all2graph.macro import CREATED_TIME


def test_descrete():
    array = ['a', 'a', 'b', 'c', None, np.nan]
    discrete = Discrete.from_data(array)
    assert abs(sum(discrete.prob.values()) - 1) < 1e-5, '概率之和不为1'
    assert abs(discrete.prob['a'] - 1/3) < 1e-5
    assert abs(discrete.prob['b'] - 1/6) < 1e-5
    assert abs(discrete.prob['c'] - 1/6) < 1e-5
    assert abs(discrete.prob[None] - 1/3) < 1e-5
    assert discrete.num_samples == 6

    json_obj = discrete.to_json()
    discrete = Discrete.from_json(json_obj)
    assert json_obj == discrete.to_json()
    print(json_obj)


def test_merge():
    arrays = []
    discretes = []
    for i in range(1, 100):
        array = np.random.choice(['a', 'b', 'c', None], size=i, replace=True)
        discrete = Discrete.from_data(array)
        arrays.append(array)
        discretes.append(discrete)

    discrete1 = Discrete.merge(discretes)
    discrete2 = Discrete.from_data(np.concatenate(arrays))
    assert {
        k: v for k, v in discrete1.to_json().items() if k != CREATED_TIME
    } == {
        k: v for k, v in discrete2.to_json().items() if k != CREATED_TIME
    }, '{}\n{}'.format(discrete1.to_json(), discrete2.to_json())
    print(discrete1.to_json())


if __name__ == '__main__':
    test_descrete()
    test_merge()
    print('测试离散分布成功')
