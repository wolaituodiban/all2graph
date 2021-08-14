import json
import os

import numpy as np
import pandas as pd

from all2graph.meta_graph.meta_node import StringNode
from toad.utils.progress import Progress
from tqdm import tqdm


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
    cat = StringNode.from_data(df['id'].unique().shape[0], df['id'], df['value'])
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
    cat1 = StringNode.from_data(df1['id'].unique().shape[0], df1['id'], df1['value'])

    df2 = pd.DataFrame(
        [
            [1, 'a'],
            [1, 'a'],
            [2, 'a'],
            [2, 'a']
        ],
        columns=['id', 'value']
    )
    cat2 = StringNode.from_data(df2['id'].unique().shape[0], df2['id'], df2['value'])
    assert cat1 != cat2
    print('test_not_eq成功')


def test_merge():
    dfs = []
    cats = []
    for i in range(1, 100):
        index = np.random.choice(list(range(10*i, 10*(i+1))), 10, replace=True)
        value = np.random.choice(['a', 'b', 'c'], 10, replace=True)

        df = pd.DataFrame({'index': index, 'value': value})
        cat = StringNode.from_data(df.shape[0], df['index'], df['value'])
        cat2 = StringNode.from_json(json.dumps(cat.to_json()))
        assert cat == cat2, '{}\n{}'.format(
            cat.to_json(), cat2.to_json()
        )

        dfs.append(df)
        cats.append(cat)

    df = pd.concat(dfs)
    cat1 = StringNode.from_data(df.shape[0], df['index'], df['value'])
    cat2 = StringNode.reduce(cats)
    assert cat1 == cat2, '{}\n{}'.format(cat1.to_json(), cat2.to_json())


def speed_from_data():
    print('测试from_data速度')
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.join(path, 'test_data', 'MensShoePrices', 'archive', 'train.csv')
    df = pd.read_csv(path)
    df = df.dropna(axis=1, how='all').astype(str)
    num_samples = df['id'].unique().shape[0]
    for col in Progress(df.columns[1:]):
        string_node = StringNode.from_data(num_samples, df['id'], df[col])
        print(col, len(string_node), string_node.max_len)


def speed_reduce():
    print('测试reduce速度')
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.join(path, 'test_data', 'MensShoePrices', 'archive', 'train.csv')
    string_nodes = []
    for chunk in tqdm(pd.read_csv(path, chunksize=100, nrows=1000)):
        chunk = chunk.dropna(axis=1, how='all').astype(str)
        for col in chunk.columns[1:10]:
            string_node = StringNode.from_data(chunk['id'].unique().shape[0], chunk['id'], chunk[col])
            string_nodes.append(string_node)
    import time
    start_time = time.time()
    StringNode.reduce(string_nodes)
    print(time.time() - start_time)
    # todo


if __name__ == '__main__':
    test_from_data()
    test_not_eq()
    test_merge()
    # speed_from_data()
    speed_reduce()
    print('测试Category成功')
