import all2graph as ag


def test_sorted():
    a = [{'a': 1, 'b': None}, {'b': 2}]
    op = ag.Sorted(key=['a', 'b'], reverse=True)
    assert op(a) == [{'b': 2}, {'a': 1, 'b': None}]


if __name__ == '__main__':
    test_sorted()
