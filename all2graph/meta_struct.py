import getpass
import json
import time
from abc import ABC, abstractmethod

from .macro import TYPE, VERSION, CREATED_TIME, CREATOR
from .version import __version__


class MetaStruct(ABC):
    """基类结构"""
    def __init__(self, *args, **kwargs):
        self.version = __version__
        self.created_time = time.asctime()
        self.creator = getpass.getuser()

    @abstractmethod
    def to_json(self) -> dict:
        """将对象装化成可以被json序列化的对象"""
        return {
            TYPE: self.__class__.__name__,
            VERSION: self.version,
            CREATED_TIME: self.created_time,
            CREATOR: self.creator
        }

    @classmethod
    @abstractmethod
    def from_json(cls, obj):
        if isinstance(obj, str):
            json.loads(obj)
        assert obj[TYPE] == cls.__name__, '类型错误'
        return cls(**{k: v for k, v in obj.items() if k not in (TYPE, VERSION, CREATED_TIME, CREATOR)})

    @classmethod
    @abstractmethod
    def from_array(cls, **kwargs):
        """根据向量生成元节点"""
        raise NotImplementedError
