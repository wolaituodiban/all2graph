import getpass
import json
from datetime import datetime as ddt, timezone
from abc import ABC, abstractmethod
from typing import List, Union

import pandas as pd

from .version import __version__


class MetaStruct(ABC):
    TYPE = 'type'
    VERSION = 'version'
    UPDATE_RECORDS = 'update_record'
    UPDATED_TIME = 'updated_time'
    UPDATER = 'updater'
    """基类结构"""
    def __init__(self, creator=None, update_records: List[dict] = None, initialized=False, **kwargs):
        """
        用户请使用from_json和from_data创建对象
        除非你知道你自己在干什么，否则不要调用构造函数
        :param creator: 创建者，若空，则根据update_records和系统用户名创建
        :param update_records: 包含版本，日期，修改者的三元素的列表
        :param initialized: 如果为False，那么将无法给对象增加属性
        :param kwargs:
        """
        self._initialized = initialized
        if update_records is None:
            self._update_records = [
                {
                    self.VERSION: __version__,
                    self.UPDATED_TIME: ddt.now(tz=timezone.utc).isoformat(),
                    self.UPDATER: creator or getpass.getuser()
                }
            ]
        else:
            self._update_records = update_records

    def __setattr__(self, key, value):
        if key != '_initialized':
            assert self._initialized, '不要擅自调用构造函数，请使用from_json或者from_data或者merge生成新的对象'
        super().__setattr__(key, value)

    @property
    def version(self):
        return self._update_records[-1][self.VERSION]

    @property
    def creator(self):
        return self._update_records[0][self.UPDATER]

    @property
    def created_time(self):
        return self._update_records[0][self.UPDATED_TIME]

    @property
    def updated_time(self):
        return self._update_records[-1][self.UPDATED_TIME]

    @abstractmethod
    def __eq__(self, other) -> bool:
        return type(self) == type(other)

    @abstractmethod
    def to_json(self) -> dict:
        """将对象装化成可以被json序列化的对象"""
        return {
            self.TYPE: self.__class__.__name__,
            self.UPDATE_RECORDS: self._update_records
        }

    @classmethod
    @abstractmethod
    def from_json(cls, obj: Union[str, dict]):
        if isinstance(obj, str):
            obj = json.loads(obj)
        return cls(initialized=True, **{k: v for k, v in obj.items() if k != cls.TYPE})

    @classmethod
    @abstractmethod
    def from_data(cls, **kwargs):
        """根据向量生成元节点"""
        return cls(initialized=True, **kwargs)

    @classmethod
    @abstractmethod
    def reduce(cls, structs, **kwargs):
        """
        合并多个结构，返回一个加总的结构
        会自动解析update_records，并生成一个合并后的update_records
        """
        update_records = []
        for struct in structs:
            update_records += struct._update_records
        update_records = pd.DataFrame(update_records)
        update_records = update_records.sort_values(cls.UPDATED_TIME)

        new_update_records = [update_records.iloc[0].to_dict()]
        # 将时间戳转换成日期，并根据日期进行去重去重
        update_records['date'] = update_records[cls.UPDATED_TIME].str[:10]
        update_records = update_records.iloc[1:].drop_duplicates('date', keep='last')
        update_records = update_records.drop(columns='date')
        new_update_records += update_records.to_dict(orient='records')

        return cls(update_records=new_update_records, initialized=True, **kwargs)
