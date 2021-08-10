from .meta_node import MetaNode
from ..stats import ECDF


class BoolNode(MetaNode):
    def __init__(self, node_freq: ECDF, value_dist: float, **kwargs):
        """
        :params node_freq:
        :params value_dist: float, True的概率
        """
        super().__init__(node_freq=node_freq, value_dist=value_dist, **kwargs)

    @classmethod
    def 
