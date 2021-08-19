import json
import os
import time

import numpy as np
import pandas as pd

from all2graph.meta_graph import MetaString
from all2graph.resolver import JsonResolver
from toad.utils.progress import Progress


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
    cat = MetaString.from_data(df['id'].unique().shape[0], df['id'], df['value'])
    assert cat['a'].mean_var == (0.5, 0.25), '{}:{}'.format(cat['a'].to_json(), cat['a'].mean_var)
    assert cat['b'].mean_var == (1.5, 0.25), '{}'.format(cat['b'].mean_var)


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
    cat1 = MetaString.from_data(df1['id'].unique().shape[0], df1['id'], df1['value'])

    df2 = pd.DataFrame(
        [
            [1, 'a'],
            [1, 'a'],
            [2, 'a'],
            [2, 'a']
        ],
        columns=['id', 'value']
    )
    cat2 = MetaString.from_data(df2['id'].unique().shape[0], df2['id'], df2['value'])
    assert cat1 != cat2


def test_merge():
    dfs = []
    cats = []
    weights = []
    for i in range(1, 100):
        index = np.random.randint(3*(i-1), 3*i, 10)
        value = np.random.choice(['a', 'b', 'c'], 10, replace=True)

        df = pd.DataFrame({'index': index, 'value': value})
        cat = MetaString.from_data(df.shape[0], df['index'], df['value'])
        cat2 = MetaString.from_json(json.loads(json.dumps(cat.to_json())))
        assert cat == cat2, '{}\n{}'.format(
            cat.to_json(), cat2.to_json()
        )

        dfs.append(df)
        cats.append(cat)
        weights.append(df.shape[0])

    df = pd.concat(dfs)
    cat1 = MetaString.from_data(df.shape[0], df['index'], df['value'])
    cat2 = MetaString.reduce(cats, weights=weights)
    assert cat1.freq == cat2.freq, '{}\n{}'.format(cat1.freq.probs, cat2.freq.probs)
    for k in cat1:
        assert cat1[k] == cat2[k], '{}\n{}\n{}'.format(k, cat1[k].quantiles, cat2[k].quantiles)


def speed():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    df = pd.read_csv(path)
    json_graph = JsonResolver(flatten_dict=True).resolve('graph', list(map(json.loads, df.json)))

    num_samples = json_graph.num_components

    groups = []
    node_df = json_graph.node_df()
    for name, group in node_df.groupby('name'):
        group['value'] = group['value'].astype(str)
        if group['value'].notna().any() and group['value'].unique().shape[0] < 1000:
            groups.append(group)

    merge_start_time = time.time()
    meta_strings = [
        MetaString.from_data(
            num_samples=num_samples, sample_ids=group.component_id, values=group.value, num_bins=20
        )
        for group in Progress(groups)
    ]
    merge_time = time.time() - merge_start_time

    reduce_start_time = time.time()
    meta_string = MetaString.reduce(meta_strings, num_bins=20)
    reduce_time = time.time() - reduce_start_time

    print(reduce_time, merge_time)
    print(meta_string.freq.quantiles)
    print(meta_string.freq.probs)
    print(meta_string.freq.get_quantiles([0.25, 0.5, 0.75], fill_value=(0, np.inf)))
    print(len(meta_string))
    print(sum(meta_string.to_discrete().prob.values()))
    assert reduce_time < merge_time


if __name__ == '__main__':
    test_from_data()
    test_not_eq()
    test_merge()
    speed()
