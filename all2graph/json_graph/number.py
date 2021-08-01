from typing import Tuple, List
from all2graph.meta_graph import MetaNode


class Number(MetaNode):
    """数值型节点"""
    def __init__(self, name: str, ecdf: Tuple[List[float], List[float]], **kwargs):
        """
        {}:param edf: 经验累计分布函数，输入的格式为([x_1, x_2, ..., x_n, n_n+1], [y_1, y_2, ..., y_n, y_n]),
            代表的分布函数等价于
                   |   0, x < x_1
            F(x) = | y_k, x_k <= x < x_k+1
                   |   1, x_n+1 <= x
        """
        super().__init__(name, **kwargs)
        assert len(ecdf[0]) == len(ecdf[1]) + 1, 'edf的第一个list的长度必须是第二个list的长度加1'
        assert min(ecdf[1]) > 0 and max(ecdf[1]) < 1, 'edf的第二个list的取值必须在0到1之间'
        self.ecdf = ecdf

    __init__.__doc__ = __init__.__doc__.format(MetaNode.__init__.__doc__)

    def to_json(self) -> dict:
        output = super().to_json()
        output['ecdf'] = self.ecdf
        return output

    @property
    def mean(self) -> float:
        mean = 0
        for x, p1, p2 in zip(self.ecdf[0], [0.]+self.ecdf[1], self.ecdf[1]+[1.]):
            mean += x * (p2 - p1)
        return mean

    @classmethod
    def from_array(cls, x, *args, **kwargs):
        pass
