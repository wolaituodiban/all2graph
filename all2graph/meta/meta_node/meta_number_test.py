import os
import json
import time
import numpy as np
import pandas as pd
from all2graph import MetaNumber
from all2graph.json import JsonParser
from toad.utils.progress import Progress


def speed():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    df = pd.read_csv(path)
    json_graph, *_ = JsonParser('graph', flatten_dict=True).parse(list(map(json.loads, df.json)))

    num_samples = json_graph.num_components

    groups = []
    node_df = pd.DataFrame(
        {
            'component_id': json_graph.component_ids,
            'name': json_graph.names,
            'value': json_graph.values,
        }
    )
    for name, group in node_df.groupby('name'):
        group['value'] = pd.to_numeric(group['value'], errors='coerce')
        if group['value'].notna().any():
            groups.append(group)

    merge_start_time = time.time()
    numbers = [
        MetaNumber.from_data(
            num_samples=num_samples, sample_ids=group.component_id, values=group.value, num_bins=20
        )
        for group in Progress(groups)
    ]
    merge_time = time.time() - merge_start_time

    reduce_start_time = time.time()
    number = MetaNumber.reduce(numbers, num_bins=20)
    reduce_time = time.time() - reduce_start_time

    assert reduce_time < merge_time
    print(reduce_time, merge_time)
    print(number.count_ecdf.quantiles)
    print(number.count_ecdf.probs)
    print(number.count_ecdf.get_quantiles([0.25, 0.5, 0.75], fill_value=(0, np.inf)))
    print(number.value_ecdf.get_quantiles([0.25, 0.5, 0.75]))


if __name__ == '__main__':
    speed()
