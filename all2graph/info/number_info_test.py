import numpy as np
import all2graph as ag


def test_from_data():
    counts = [0, 3, 2]
    values = [0, 1, 1, 2]
    number_info = ag.NumberInfo.from_data(counts=counts, values=values)
    print(number_info)


def test_zero_count():
    number_info = ag.NumberInfo.from_data(counts=[0], values=[])
    print(number_info)


def test_reduce():
    counts = []
    values = []
    number_infos = []
    for _ in range(200):
        count = np.random.randint(10)
        value = np.random.randint(low=0, high=5, size=count)
        number_info = ag.NumberInfo.from_data(counts=[count], values=value)
        counts.append(count)
        values.append(value)
        number_infos.append(number_info)
    values = np.concatenate(values)
    number_info1 = ag.NumberInfo.reduce(number_infos)
    number_info2 = ag.NumberInfo.from_data(counts=counts, values=values)
    assert number_info1 == number_info2, (number_info1.value.quantiles, number_info2.value.quantiles)


if __name__ == '__main__':
    test_from_data()
    test_zero_count()
    test_reduce()
