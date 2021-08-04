import getpass
import json
from datetime import datetime as ddt, timezone
from abc import ABC, abstractmethod

from .macro import TYPE, VERSION, CREATED_TIME, CREATOR
from .version import __version__


class MetaStruct(ABC):
    """基类结构"""
    def __init__(self, created_time=None, creator=None, **kwargs):
        self.version = __version__
        self.created_time = created_time or ddt.now(tz=timezone.utc).isoformat()
        self.creator = creator or getpass.getuser()

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
        return cls(**{k: v for k, v in obj.items() if k not in (TYPE, VERSION)})

    @classmethod
    @abstractmethod
    def from_data(cls, **kwargs):
        """根据向量生成元节点"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def merge(cls, **kwargs):
        """合并多个经验累计分布函数，返回一个贾总的经验累计分布函数"""
        raise NotImplementedError
