from .operator import Operator


class Delete(Operator):
    def __init__(self, names):
        assert isinstance(names, (list, tuple))
        super().__init__()
        self.names = names

    def __call__(self, obj, **kwargs):
        for name in self.names:
            if name in obj:
                del obj[name]
        return obj

    def __repr__(self):
        return '{}(names={})'.format(self.__class__.__name__, self.names)
