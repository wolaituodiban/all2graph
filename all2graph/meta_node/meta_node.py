from abc import abstractmethod

from ..meta_struct import MetaStruct


class MetaNode(MetaStruct):
    """
    节点的基类，定义基本成员变量和基本方法

    在all2graph的视角下，有两个尺度的观察口径
    1、样本口径，每一个小图被称为样本
    2、节点口径，小图中的每一个点被称为节点

    于是，看待数据分布时，有两个不同的统计口径
    1、样本的口径
    2、节点的口径

    对于不同类型的节点，其统计分布的口径会有不同，需要区分对待
    """
    pass

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        raise NotImplementedError
