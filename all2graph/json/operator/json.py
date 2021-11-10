import json as py_json

from typing import Set

from .operator import Operator
from ..utils import unstructurize_dict


class JsonDumper(Operator):
    def __init__(self, error=False):
        super().__init__()
        self.error = error

    def __call__(self, obj, **kwargs):
        try:
            return py_json.loads(obj)
        except py_json.JSONDecodeError as e:
            if self.error:
                raise e
        return obj

    def __repr__(self):
        return '{}(error={})'.format(self.__class__.__name__, self.error)


class Unstructurizer(Operator):
    def __init__(self, preserved: Set[str] = None, start_level=0, end_level=0):
        super().__init__()
        self.preserved = preserved
        self.start_level = start_level
        self.end_level = end_level

    def __call__(self, obj, **kwargs):
        return unstructurize_dict(obj, preserved=self.preserved, start_level=self.start_level, end_level=self.end_level)

    def __repr__(self):
        return '{}(preserved={}, start_level={}, end_level={})'.format(
            self.__class__.__name__, self.preserved, self.start_level, self.end_level)


class DictGetter(Operator):
    def __init__(self, keys: Set[str]):
        super(DictGetter, self).__init__()
        self.keys = keys

    def __call__(self, obj, **kwargs):
        return {k: obj[k] for k in self.keys if k in obj}

    def __repr__(self):
        return '{}(keys={})'.format(
            self.__class__.__name__, self.keys)


class ConcatList(Operator):
    def __init__(self, inputs: set, output: str):
        super(ConcatList, self).__init__()
        assert isinstance(inputs, (list, tuple, set))
        self.inputs = set(inputs)
        self.output = output

    def __call__(self, obj, **kwargs):
        output = {}
        for k, v in obj.items():
            if k in self.inputs:
                output[self.output] = output.get(self.output, []) + v
            else:
                output[k] = v
        return output
