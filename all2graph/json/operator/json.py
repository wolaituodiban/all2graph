import json as py_json
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
    def __call__(self, obj, **kwargs):
        return unstructurize_dict(obj)

