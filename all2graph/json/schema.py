from typing import Dict, Any

from ..meta_struct import MetaStruct
from ..globals import EPSILON


def _json_flat(obj, output: Dict[str, Any], path: str) -> Dict[str, Any]:
    output[path] = obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            _json_flat(v, output=output, path='.'.join([path, k]))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _json_flat(item, output=output, path=path+'[{}]'.format(i))
    return output


def json_flat(obj) -> Dict[str, Any]:
    """

    Args:
        obj: json object

    Returns:
        a dict with key is the json path and value is the sub-object
    """
    return _json_flat(obj, output={}, path='$')


def _json_diff(a, b):
    diff = {}
    for right_missing in set(a).difference(b):
        diff[right_missing] = {'type': 'missing', 'how': 'right'}

    for left_missing in set(b).difference(a):
        diff[left_missing] = {'type': 'missing', 'how': 'left'}

    for common in set(a).intersection(b):
        a_value = a[common]
        b_value = b[common]
        if not isinstance(a_value, type(b_value)) or not isinstance(b_value, type(a_value)):
            diff[common] = {'type': 'diff type', 'left': str(type(a_value)), 'right': str(type(b_value))}
        elif isinstance(a_value, (list, dict)):
            continue

        try:
            a_value, b_value = float(a_value), float(b_value)
        except (ValueError, TypeError):
            pass

        if (isinstance(a_value, (int, float)) and abs(a_value - b_value) > EPSILON) or (a_value != b_value):
            diff[common] = {'type': 'diff value', 'left': a_value, 'right': b_value}
    return diff


def json_diff(a, b):
    """

    Args:
        a: left json object
        b: right json object

    Returns:
        a dict with key the json path of different part
        and value the description of difference
    """
    return _json_diff(json_flat(a), json_flat(b))


def json_schema(obj):
    """

    Args:
        obj (dict, list): json object

    Returns:
        dict: key is json path, value is set of type
    """
    def add_type(_type):
        if path in output:
            output[path].add(_type)
        else:
            output[path] = {_type}

    output = {}
    for path, obj in json_flat(obj).items():
        if isinstance(obj, dict):
            add_type('dict')
        elif isinstance(obj, list):
            add_type('list')
        elif isinstance(obj, str):
            add_type('str')
        elif isinstance(obj, int):
            add_type('int')
        elif isinstance(obj, float):
            add_type('float')
        elif isinstance(obj, bool):
            add_type('bool')
        elif obj is None:
            add_type(None)
        else:
            add_type('unknown')
    return output


class JsonSchema(MetaStruct):
    def __init__(self, obj):
        super().__init__(initialized=True)
        self.schema = json_schema(obj)

    def __eq__(self, other):
        return len(_json_diff(self.schema, other.schema)) == 0

    def to_json(self) -> dict:
        return self.schema

    @classmethod
    def from_json(cls, obj: dict):
        output = cls({})
        output.schema = obj
        return output

    @classmethod
    def from_data(cls, obj, **kwargs):
        return cls(obj)

    @classmethod
    def reduce(cls, structs, weights=None, **kwargs):
        output = cls({})
        for struct in structs:
            for path, types in struct.schema.items():
                if path in output.schema:
                    output.schema[path] = output.schema[path].union(types)
                else:
                    output.schema[path] = types
        return output

    def diff(self, obj):
        return _json_diff(self.schema, json_schema(obj))

    def validate(self, obj, strict=True):
        schema_diff = self.diff(obj)
        if not strict:
            schema_diff = {k: v for k, v in schema_diff.items() if v['type'] != 'missing' or v['how'] != 'right'}
        return len(schema_diff) == 0


