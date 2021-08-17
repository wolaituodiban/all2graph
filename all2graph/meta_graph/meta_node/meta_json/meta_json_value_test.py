import json
import os
import pandas as pd
from toad.utils.progress import Progress
from all2graph.meta_graph.meta_node import MetaJsonValue


def test_json_value():
    sample_ids1 = [0, 0, 1, 1, 1, 2, 3, 3, 4]
    values1 = ['a', 'b', 1, '2020-01-01', 3, 'c', True, False, '2022-01-23 12:34:01']
    sample_times1 = ['2020-01-08'] * len(sample_ids1)

    jv1 = MetaJsonValue.from_data(len(sample_ids1), sample_ids1, values1, sample_times=sample_times1)

    sample_ids2 = [5, 5, 6, 7, 8, 8, 8, 9, 10, 10, 11]
    values2 = [3, 'b', 1, '2020-01-01', 3, 'c', True, False, '2022-01-23 12:34:01', 'b', None]
    sample_times2 = ['2020-01-08'] * len(sample_ids2)

    jv2 = MetaJsonValue.from_data(len(sample_ids2), sample_ids2, values2, sample_times=sample_times2)

    sample_ids3 = sample_ids1 + sample_ids2
    values3 = values1 + values2
    sample_times3 = sample_times1 + sample_times2

    jv3 = MetaJsonValue.from_data(len(sample_ids3), sample_ids3, values3, sample_times=sample_times3)
    jv4 = MetaJsonValue.reduce([jv1, jv2])

    assert jv3.freq == jv4.freq, '{}\n{}'.format(jv3.freq.to_json(), jv4.freq.to_json())
    for k in jv3.meta_data:
        if isinstance(jv3.meta_data[k].meta_data, dict):
            a = jv3.meta_data[k].meta_data
            b = jv4.meta_data[k].meta_data
            for kk in a:
                assert a[kk] == b[kk], '{}\n{}\n{}'.format(
                    kk, a[kk].to_json(), b[kk].to_json()
                )
        else:
            assert jv3.meta_data[k] == jv4.meta_data[k], '{}\n{}'.format(
                jv3.meta_data[k].to_json(), jv4.meta_data[k].to_json()
            )
    for k in jv3.meta_data:
        assert jv3.meta_data[k] == jv4.meta_data[k], '{}\n{}'.format(
            jv3.meta_data[k].to_json(), jv4.meta_data[k].to_json()
        )

    jv5 = jv3.from_json(json.dumps(jv3.to_json()))
    print(jv5.to_discrete().to_json())
    assert jv3 == jv5


def speed():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.join(path, 'test_data', 'MensShoePrices', 'archive', 'train.csv')
    df = pd.read_csv(path)
    num_samples = df['id'].unique().shape[0]
    for col in Progress(df.drop(columns=['id', 'dateadded']).columns):
        json_value = MetaJsonValue.from_data(num_samples, df['id'], df[col], sample_times=df.dateadded)
        print(col, json_value.to_discrete().prob)


if __name__ == '__main__':
    test_json_value()
    speed()
    print('测试JsonValue成功')
