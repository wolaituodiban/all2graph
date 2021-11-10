import json
import all2graph as ag


def test_json_schema():
    obj1 = {'a': [1, None, {'b': 'c'}], 'b': 2}
    obj2 = {'a': [1], 'b': 2}
    obj3 = {'a': ['str']}
    obj4 = {'c': 3}

    schema = ag.JsonSchema(obj1)
    assert schema.validate(obj2, strict=False), schema.diff(obj2, strict=False)
    assert not schema.validate(obj2)

    assert not schema.validate(obj3, strict=False)
    assert not schema.validate(obj3)

    assert not schema.validate(obj4, strict=False)
    assert not schema.validate(obj4, strict=True)


def test_to_json():
    obj1 = {'a': [1, None, {'b': 'c'}], 'b': 2}
    schema = ag.JsonSchema(obj1)
    json.dumps(schema.to_json())


if __name__ == '__main__':
    test_json_schema()
    test_to_json()
