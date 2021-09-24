from all2graph.version import __version__


class Operator:
    def __init__(self):
        # todo 增加一个check 数据的功能，选择是否raise Exception
        self.version = __version__

    def __call__(self, obj, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)








