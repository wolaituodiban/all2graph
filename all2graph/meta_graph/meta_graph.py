from abc import ABC, abstractmethod
import getpass
import time
from typing import List

import networkx as nx

from all2graph.macro import TYPE
from all2graph.version import __version__
from .meta_node import MetaNode


class MetaGraph(ABC):
    """图的基类，定义基本成员变量和基本方法"""
    def __init__(self, nodes: List[MetaNode]):
        """

        :param nodes: 节点列表
        """
        self.version = __version__
        self.created_time = time.asctime()
        self.updated_time = self.created_time
        self.creator = getpass.getuser()

        # 检查是否存在命名冲突
        assert len(nodes) == len(set(node.name for node in nodes)), '存在节点命名冲突'
        self.nodes = list(nodes)
        # 检查是否存在没有属性的节点
        graph = self.to_networkx()
        assert graph.number_of_nodes() == len(nodes), '请检查每个节点的每一个前置节点和后置节点，是否都在nodes中'

    @abstractmethod
    def to_json(self) -> dict:
        """将对象转化成一个可以被json序列化的对象"""
        return {
            TYPE: self.__class__.__name__,
            'created_time': self.created_time,
            'updated_time': self.updated_time,
            'creator': self.creator,
            'nodes': [node.to_json() for node in self.nodes]
        }

    def to_networkx(self) -> nx.DiGraph:
        """将对象转化成一个networkx有向图"""
        graph = nx.DiGraph()
        for node in self.nodes:
            graph.add_node(node.name, **node.to_json())
            for pred in node.preds:
                graph.add_edge(pred, node.name)
            for succ in node.succs:
                graph.add_edge(node.name, succ)
        return graph
