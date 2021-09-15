from all2graph.version import __version__


class Operator:
    def __init__(self):
        self.version = __version__

    def __call__(self, obj, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)








