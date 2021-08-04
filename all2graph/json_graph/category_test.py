import numpy as np
from all2graph.json_graph import Category


def test_category():
    array = ['a', 'a', 'b', 'c', None, np.nan]
    category = Category.from_data(array)
    assert abs(sum(category.frequency.values()) - 1) < 1e-5, '概率之和不为1'
    assert abs(category['a'] - 1/3) < 1e-5
    assert abs(category['b'] - 1/6) < 1e-5
    assert abs(category['c'] - 1/6) < 1e-5
    assert abs(category[None] - 1/3) < 1e-5
    assert category.num_samples == 6

    json_obj = category.to_json()
    category = Category.from_json(json_obj)
    assert json_obj == category.to_json()
    print(json_obj)


if __name__ == '__main__':
    test_category()
    print('测试类别节点成功')
