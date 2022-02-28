import all2graph as ag
import numpy as np


def test_from_data():
    counts = np.random.randint(0, 100, size=100)
    num_nodes = counts + np.random.randint(0, 100, size=100)
    token_info = ag.TokenInfo.from_data(counts=counts, num_nodes=num_nodes)
    print(token_info)


def test_reduce():
    weights = []
    counts = []
    num_nodes = []
    infos = []
    for i in range(20):
        weight = np.random.randint(1, 100)
        _counts = np.random.randint(0, 100, size=weight)
        _num_nodes = _counts + np.random.randint(0, 100, size=weight)
        _token_info = ag.TokenInfo.from_data(counts=_counts, num_nodes=_num_nodes)
        weights.append(weight)
        counts.append(_counts)
        num_nodes.append(_num_nodes)
        infos.append(_token_info)

    info1 = ag.TokenInfo.reduce(infos, weights=weights)

    counts = np.concatenate(counts)
    num_nodes = np.concatenate(num_nodes)
    info2 = ag.TokenInfo.from_data(counts=counts, num_nodes=num_nodes)
    assert info1.__eq__(info2, debug=True), (info1, info2)


if __name__ == '__main__':
    test_from_data()
    test_reduce()
