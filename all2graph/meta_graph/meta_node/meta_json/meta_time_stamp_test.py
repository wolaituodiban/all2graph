import json
import os
import time

import numpy as np
import pandas as pd

from all2graph.graph import JsonGraph
from all2graph.meta_graph import MetaTimeStamp
from toad.utils.progress import Progress


def test_timestamp():
    a1 = ['2020-01-02', '2020-05-01 12:33:11']
    a2 = ['2020-01-02', '2020-05-03 12:01:23.11']
    sample_time = '2021-05-01'

    t1 = MetaTimeStamp.from_data(len(a1), sample_ids=list(range(len(a1))), values=a1, sample_times=sample_time)
    t2 = MetaTimeStamp.from_data(len(a2), sample_ids=list(range(len(a2))), values=a2, sample_times=sample_time)
    assert t1 != t2

    t3 = MetaTimeStamp.from_json(json.loads(json.dumps(t1.to_json())))
    assert t1 == t3, '{}\n{}'.format(t1.to_json(), t3.to_json())
    a3 = a1 + a2
    t4 = MetaTimeStamp.from_data(len(a3), sample_ids=list(range(len(a3))), values=a3, sample_times=sample_time)
    t5 = MetaTimeStamp.reduce([t1, t2])
    assert t4 == t5
    print(t5.to_discrete().to_json())


def speed():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    df = pd.read_csv(path)
    json_graph = JsonGraph(flatten_dict=True)
    for i, value in enumerate(Progress(df.json.values)):
        json_graph.insert_component(i, 'graph', json.loads(value))

    num_samples = json_graph.num_components

    groups = []
    node_df = json_graph.nodes_to_df()
    for name, group in Progress(node_df.groupby('name')):
        group['value'] = group['value'].apply(lambda x: None if isinstance(x, (dict, list)) else x)
        group['value'] = pd.to_datetime(group['value'], errors='coerce')
        if group['value'].notna().any():
            groups.append(group)

    merge_start_time = time.time()
    timestamps = [
        MetaTimeStamp.from_data(
            num_samples=num_samples, sample_ids=group.component_id, values=group.value, sample_times='2021-02-01',
            num_bins=20
        )
        for group in Progress(groups)
    ]
    merge_time = time.time() - merge_start_time

    reduce_start_time = time.time()
    timestamp = MetaTimeStamp.reduce(timestamps, num_bins=20)
    reduce_time = time.time() - reduce_start_time

    print(reduce_time, merge_time)
    print(timestamp.freq.quantiles)
    print(timestamp.freq.probs)
    print(timestamp.freq.get_quantiles([0.25, 0.5, 0.75], fill_value=(0, np.inf)))
    print(timestamp.meta_data.keys())
    assert reduce_time < merge_time


if __name__ == '__main__':
    test_timestamp()
    speed()
