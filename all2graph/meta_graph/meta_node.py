from abc import ABC, abstractmethod
from typing import List
from all2graph.macro import TYPE, NAME, PREDS, SUCCS


class MetaNode(ABC):
    """节点的基类，定义基本成员变量和基本方法"""
    def __init__(self, name: str, preds: List[str] = None, succs: List[str] = None):
        """

        :param name: 节点名字
        :param preds: 前置节点的名字
        :param succs: 后置节点的名字
        """
        self.name = name
        self.preds = list(preds or [])
        self.succs = list(succs or [])

    @abstractmethod
    def to_json(self) -> dict:
        """将对象装化成可以被json序列化的对象"""
        return {
            TYPE: self.__class__.__name__,
            NAME: self.name,
            PREDS: self.preds,
            SUCCS: self.succs
        }

    @classmethod
    @abstractmethod
    def from_array(cls, x, *args, **kwargs):
        """
        根据向量生成元节点
        :param x: 向量
        :param args: 构造函数的参数
        :param kwargs: 构造函数的参数
        :return:
        """
        raise NotImplementedError
