import json

import numpy as np
import pandas as pd

from all2graph.node import Category


def test_from_data():
    df = pd.DataFrame(
        [
            [1, 'a'],
            [1, 'b'],
            [1, 'b'],
            [2, 'b']
        ],
        columns=['id', 'value']
    )
    cat = Category.from_data(df, id_col='id', value_col='value')
    assert cat['a'].mean_var == (0.5, 0.25), '{}'.format(cat['a'].mean_var)
    assert cat['b'].mean_var == (1.5, 0.25), '{}'.format(cat['b'].mean_var)
    print('测试from_date成功')


def test_none():
    df = pd.DataFrame(
        [
            [1, None],
            [1, None],
            [1, None],
            [2, None]
        ],
        columns=['id', 'value']
    )
    cat = Category.from_data(df, id_col='id', value_col='value')
    print(cat.to_json())
    print('测试test_none成功')


def test_merge():
    dfs = []
    cats = []
    for i in range(1, 100):
        index = np.random.choice(list(range(10*i, 10*(i+1))), 10, replace=True)
        value = np.random.choice(['a', 'b', 'c', None], 10, replace=True)

        df = pd.DataFrame({'index': index, 'value': value})
        cat = Category.from_data(df, id_col='index', value_col='value')

        assert cat.to_json() == Category.from_json(json.dumps(cat.to_json())).to_json(), '{}\n{}'.format(
            cat.to_json(), Category.from_json(json.dumps(cat.to_json())).to_json()
        )

        dfs.append(df)
        cats.append(cat)

    df = pd.concat(dfs)
    cat1 = Category.from_data(df, id_col='index', value_col='value')
    cat2 = Category.merge(cats)
    assert cat1.to_json() == cat2.to_json(), '{}\n{}'.format(cat1.to_json(), cat2.to_json())


if __name__ == '__main__':
    test_from_data()
    test_none()
    test_merge()
    print('测试Category成功')
