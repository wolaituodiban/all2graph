from .operator import Operator


class Sorted(Operator):
    def __init__(self, key=None, reverse=False):
        super().__init__()
        self.key = key
        self.reverse = reverse

    def __call__(self, obj, **kwargs):
        if self.key is None:
            return sorted(obj, key=lambda x: x[self.key] if self.key in x else None, reverse=self.reverse)
        else:
            return sorted(obj, reverse=self.reverse)