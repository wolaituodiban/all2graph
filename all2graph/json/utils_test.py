import all2graph as ag


def test_unstructurize_dict1():
    a = {
        'someThing': 1,
        'some_key': 2,
        'key': 3
    }
    b = ag.unstructurize_dict(a)

    assert b == {'some': {'thing': 1, 'key': 2}, 'key': 3}, b


def test_unstructurize_dict2():
    a = {
        'someThingHa': 1,
        'some_key_ha': 2,
        'key': 3
    }

    b = ag.unstructurize_dict(a, start_level=1, end_level=1)

    assert b == {'thing': 1, 'key': 2}, b


def test_unstructurize_dict3():
    a = {
        'someThingHa': 1,
        'some_key_ha': 2,
        'key': 3
    }

    b = ag.unstructurize_dict(a, preserved={'key'}, start_level=3, end_level=1)

    assert b == {'key': 3}, b


if __name__ == '__main__':
    test_unstructurize_dict1()
    test_unstructurize_dict2()
    test_unstructurize_dict3()
