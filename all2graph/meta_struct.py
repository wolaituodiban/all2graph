import sys
from abc import ABC

import numpy as np

from .version import __version__


class MetaStruct(ABC):
    """基类结构"""
    def __init__(self, initialized=False, type=None, version=None):
        """
        用户请使用from_json和from_data创建对象
        除非你知道你自己在干什么，否则不要调用构造函数
        :param.py initialized: 如果为False，那么将无法给对象增加属性
        :param.py kwargs:
        """
        if type is not None:
            assert type == self.__class__.__name__, 'type不匹配'
        if version is not None and version != __version__:
            print('{}.__init__ current version is {}, but got version {}'.format(
                self.__class__.__name__, __version__, version), file=sys.stderr)
        self._initialized = initialized
        self.version = __version__

    def __setattr__(self, key, value):
        if key != '_initialized':
            assert self._initialized, '不要擅自调用构造函数，请使用from_json或者from_data或者merge生成新的对象'
        super().__setattr__(key, value)

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
        return cls(initialized=True, **obj)

    @classmethod
    def from_data(cls, **kwargs):
        """根据向量生成元节点"""
        return cls(initialized=True, **kwargs)

    @classmethod
    def batch(cls, structs, weights=None, **kwargs):
        """
        合并多个结构，返回一个加总的结构
        """
        return cls(initialized=True, **kwargs)

    def extra_repr(self) -> str:
        return ''

    def __repr__(self):
        extra_repr = str(self.extra_repr())
        if '\n' in extra_repr:
            extra_repr = '\n  ' + '\n  '.join(extra_repr.split('\n')) + '\n'
        return '{}({})'.format(self.__class__.__name__, extra_repr)


def equal(a, b, prefix=''):
    for k, v in a.__dict__.items():
        print(prefix, k)
        if k not in b.__dict__:
            print(prefix, 'not in b')
            return False
        v2 = b.__dict__[k]
        if isinstance(v, list):
            for i, vv in enumerate(v):
                print(prefix, i)
                if not equal(vv, v2[i], prefix+' '):
                    return False
        elif isinstance(v, dict):
            for kk, vv in v.items():
                print(prefix, kk)
                if not equal(vv, v2[kk], prefix+' '):
                    return False
        elif isinstance(v, (float, int, bool, np.ndarray)):
            if not np.allclose(v, v2):
                print(prefix, 'not equal')
                return False
        elif isinstance(v, MetaStruct):
            if not equal(v, v2, prefix+' '):
                print(prefix, 'not equal')
                return False
        elif v != v2:
            print(prefix, 'not equal')
            return False
    return True
