import json
from .operator import Operator


class JsonDumper(Operator):
    def __init__(self, error=False):
        super().__init__()
        self.error = error

    def __call__(self, obj, **kwargs):
        try:
            return json.loads(obj)
        except json.JSONDecodeError as e:
            if self.error:
                raise e
        return obj

    def __repr__(self):
        return '{}(error={})'.format(self.__class__.__name__, self.error)
