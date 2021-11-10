import all2graph as ag


def test_unstructurizor():
    a = {
        'someThingHa': 1,
        'some_like_ha': 2,
        'key': 3
    }
    op = ag.Unstructurizer(preserved={'key'}, start_level=1, end_level=1)
    b = op(a)

    assert b == {'thing': 1, 'like': 2, 'key': 3}, b


def test_dict_getter():
    a = {'a': 1, 'b': 2}
    op = ag.DictGetter({'b'})
    b = op(a)
    assert b == {'b': 2}, b


def test_concat_list():
    a = {'a': [1, 2], 'b': [3]}
    op = ag.ConcatList({'a', 'b', 'c'}, 'a')
    b = op(a)
    assert a == {'a': [1, 2], 'b': [3]}
    assert b == {'a': [1, 2, 3]}


if __name__ == '__main__':
    test_unstructurizor()
    test_dict_getter()
    test_concat_list()

