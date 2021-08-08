import json

import numpy as np
import pandas as pd

from all2graph.meta_node import Category


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
    assert cat['a'].mean_var == (0.5, 0.25), '{}:{}'.format(cat['a'].to_json(), cat['a'].mean_var)
    assert cat['b'].mean_var == (1.5, 0.25), '{}'.format(cat['b'].mean_var)
    print('测试from_date成功')


def test_not_eq():
    df1 = pd.DataFrame(
        [
            [1, 'a'],
            [1, 'a'],
            [1, 'a'],
            [2, 'a']
        ],
        columns=['id', 'value']
    )
    cat1 = Category.from_data(df1, id_col='id', value_col='value')

    df2 = pd.DataFrame(
        [
            [1, 'a'],
            [1, 'a'],
            [2, 'a'],
            [2, 'a']
        ],
        columns=['id', 'value']
    )
    cat2 = Category.from_data(df2, id_col='id', value_col='value')
    assert cat1 != cat2
    print('test_not_eq成功')


def test_merge():
    dfs = []
    cats = []
    for i in range(1, 100):
        index = np.random.choice(list(range(10*i, 10*(i+1))), 10, replace=True)
        value = np.random.choice(['a', 'b', 'c', None], 10, replace=True)

        df = pd.DataFrame({'index': index, 'value': value})
        cat = Category.from_data(df, id_col='index', value_col='value')
        cat2 = Category.from_json(json.dumps(cat.to_json()))
        assert cat == cat2, '{}\n{}'.format(
            cat.to_json(), cat2.to_json()
        )

        dfs.append(df)
        cats.append(cat)

    df = pd.concat(dfs)
    cat1 = Category.from_data(df, id_col='index', value_col='value')
    cat2 = Category.merge(cats)
    assert cat1 == cat2, '{}\n{}'.format(cat1.to_json(), cat2.to_json())


if __name__ == '__main__':
    test_from_data()
    test_not_eq()
    test_merge()
    print('测试Category成功')
