import json
import os
import time

import numpy as np
import pandas as pd

from all2graph import MetaString
from all2graph.json import JsonParser
from toad.utils.progress import Progress
import json_tools


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
    print(cat)
    assert cat.term_count_ecdf['a'].mean_var == (0.5, 0.25), '{}:{}'.format(
        cat.term_count_ecdf['a'].to_json(), cat.term_count_ecdf['a'].mean_var)
    assert cat.term_count_ecdf['b'].mean_var == (1.5, 0.25), '{}'.format(
        cat.term_count_ecdf['b'].mean_var)


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
    print(cat1)

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
    print(cat2)
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
        assert cat == cat2, '\n{}\n{}'.format(
            cat.to_json(), cat2.to_json()
        )

        dfs.append(df)
        cats.append(cat)
        weights.append(df.shape[0])

    df = pd.concat(dfs)
    cat1 = MetaString.from_data(df.shape[0], df['index'], df['value'])
    cat2 = MetaString.reduce(cats, weights=weights)
    print(cat1)
    print(cat2)
    assert cat1.term_count_ecdf == cat2.term_count_ecdf, json_tools.diff(
        cat1.term_count_ecdf.to_json(), cat2.term_count_ecdf.to_json()
    )
    for k in cat1:
        assert cat1.term_freq_ecdf[k] == cat2.term_freq_ecdf[k], '{}\n{}\n{}'.format(
            k, cat1.term_freq_ecdf[k].to_json(), cat2.term_freq_ecdf[k].to_json()
        )


def speed():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    df = pd.read_csv(path)
    json_graph, *_ = JsonParser('json', flatten_dict=True).parse(df)

    num_samples = json_graph.num_components

    groups = []
    node_df = pd.DataFrame(
        {
            'component_id': json_graph.component_id,
            'name': json_graph.key,
            'value': json_graph.value,
        }
    )
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
    meta_string = MetaString.reduce(meta_strings, num_bins=20, progress_bar=True)
    reduce_time = time.time() - reduce_start_time

    print(reduce_time, merge_time)
    print(len(meta_string))
    print(sum(meta_string.to_discrete().prob.values()))
    assert reduce_time < merge_time


if __name__ == '__main__':
    test_from_data()
    test_not_eq()
    test_merge()
    speed()
