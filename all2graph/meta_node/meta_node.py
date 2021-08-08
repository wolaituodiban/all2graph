from abc import abstractmethod

from ..meta_struct import MetaStruct


class MetaNode(MetaStruct):
    """节点的基类，定义基本成员变量和基本方法"""
    pass

    @property
    @abstractmethod
    def miss_rate(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        raise NotImplementedError
