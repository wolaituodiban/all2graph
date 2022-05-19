import sys
import numpy as np
from .version import __version__

try:
    import cython
    if cython.compiled:
        print('all2graph: cython compiled', file=sys.stderr)
except ImportError:
    pass


class MetaStruct:
    """基类结构"""
    def __init__(self, type=None, version=None):
        """
        用户请使用from_json和from_data创建对象
        除非你知道你自己在干什么，否则不要调用构造函数
        :param.py initialized: 如果为False，那么将无法给对象增加属性
        :param.py kwargs:
        """
        if type is not None:
            assert type == self.__class__.__name__, 'type不匹配'
        # if version is not None and version != __version__:
        #     print('{}.__init__ current version is {}, but got version {}'.format(
        #         self.__class__.__name__, __version__, version))
        self.version = __version__

    def __eq__(self, other) -> bool:
        return self.__dict__ == other.__dict__

    def to_json(self) -> dict:
        """将对象装化成可以被json序列化的对象"""
        return {
            'type': self.__class__.__name__,
            'version': self.version
        }

    @classmethod
    def from_json(cls, obj: dict):
        return cls(**obj)

    @classmethod
    def from_data(cls, **kwargs):
        """根据向量生成元节点"""
        return cls(**kwargs)

    @classmethod
    def batch(cls, structs, weights=None, **kwargs):
        """
        合并多个结构，返回一个加总的结构
        """
        return cls(**kwargs)

    def extra_repr(self) -> str:
        return ''

    def __repr__(self):
        extra_repr = str(self.extra_repr())
        if '\n' in extra_repr:
            extra_repr = '\n  ' + '\n  '.join(extra_repr.split('\n')) + '\n'
        return '{}({})'.format(self.__class__.__name__, extra_repr)


def equal(a, b, prefix=''):
    if isinstance(a, list):
        for i, v in enumerate(a):
            print(prefix, i)
            if not equal(v, b[i], prefix + ' '):
                return False
    elif isinstance(a, dict):
        for k, v in a.items():
            print(prefix, k)
            if not equal(v, b[k], prefix + ' '):
                return False
    elif isinstance(a, (float, int, bool, np.ndarray)):
        if not np.allclose(a, b):
            print(prefix, '{} and {} not equal'.format(a, b))
            return False
    elif isinstance(a, MetaStruct):
        return equal(a.__dict__, b.__dict__, prefix + ' ')
    elif a != b:
        print(prefix, '{} and {} not equal'.format(a, b))
        return False
    return True
