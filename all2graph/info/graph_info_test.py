import numpy as np
import all2graph as ag


def test_from_data():
    sample_ids = [0, 0, 1]
    keys = ['a', ('b', 'c'), 'c']
    values = ['d', '0.2', None]
    graph_info = ag.GraphInfo.from_data(sample_ids=sample_ids, keys=keys, values=values)
    print(graph_info)


def test_reduce():
    num_nodes = 5

    sample_ids = []
    keys = []
    values = []
    graph_infos = []
    for i in range(10):
        sample_id = [i] * num_nodes
        key = np.random.choice(all_keys, size=num_nodes)
        value = np.random.choice(all_values, size=num_nodes)
        graph_info = ag.GraphInfo.from_data(sample_ids=sample_id, keys=key, values=value)

        sample_ids.append(sample_id)
        keys.append(key)
        values.append(value)
        graph_infos.append(graph_info)

    sample_ids = np.concatenate(sample_ids)
    keys = np.concatenate(keys)
    values = np.concatenate(values)

    graph_info1 = ag.GraphInfo.reduce(graph_infos)
    graph_info2 = ag.GraphInfo.from_data(sample_ids=sample_ids, keys=keys, values=values)

    assert graph_info1.__eq__(graph_info2, debug=True)
    print(graph_info1.dictionary())
    print(graph_info1.numbers)


if __name__ == '__main__':
    all_values = np.array([1, 2, 3, '1.2', 'abb', 'feisl', None], dtype=object)
    all_keys = np.array(['fbh', ('as', 'awe', 'f'), 'awef', 'fepoij', 'asdfj', 'aef'], dtype=object)

    test_from_data()
    test_reduce()
