from .operator import Operator


class Sorted(Operator):
    def __init__(self, key=None, reverse=False):
        super().__init__()
        if key:
            assert isinstance(key, (list, tuple, str))
        self.key = key
        self.reverse = reverse

    def _get_key(self, item):
        if isinstance(self.key, (list, tuple)):
            for key in self.key:
                if key in item:
                    return item[key]
        elif self.key:
            return item[self.key]

    def __call__(self, obj, **kwargs):
        if self.key is None:
            return sorted(obj, reverse=self.reverse)
        else:
            return sorted(obj, key=self._get_key, reverse=self.reverse)

    def __repr__(self):
        return '{}(key={}, reverse={})'.format(self.__class__.__name__, self.key, self.reverse)
