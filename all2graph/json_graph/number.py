from typing import Tuple, List
from all2graph.meta_graph import MetaNode


class Number(MetaNode):
    """数值型节点"""
    def __init__(self, name: str, ecdf: Tuple[List[float], List[float]], num_samples, preds: List[str] = None,
                 succs: List[str] = None):
        """

        :param name: 节点名字
        :param preds: 前置节点的名字
        :param succs: 后置节点的名字
        :param ecdf: 经验累计分布函数，输入为两个list，分别表示很坐标和纵坐标
            例如，([x_1, x_2, ..., x_n, n_n+1], [y_1, y_2, ..., y_n, y_n]),
            代表的分布函数等价于
                   |   0, x < x_1
            F(x) = | y_k, x_k <= x < x_k+1
                   |   1, x_n+1 <= x
        :param num_samples: 计算累积分布函数时使用的样本量。在合并两个节点时，会发挥作用
        """
        super().__init__(name, preds=preds, succs=succs)
        assert len(ecdf[0]) == len(ecdf[1]) + 1, 'ecdf的横坐标的长度必须是纵坐标的长度加1'
        for a, b in zip(ecdf[0][:-1], ecdf[0][1:]):
            assert a < b, 'ecdf横坐标必须是单调递增的'
        assert min(ecdf[1]) > 0 and max(ecdf[1]) < 1, 'ecdf的纵坐标的取值必须在0到1之间'
        for a, b in zip(ecdf[1][:-1], ecdf[1][1:]):
            assert a < b, 'ecdf纵坐标必须是单调递增的'
        self.ecdf = ecdf
        self.num_samples = num_samples

    @property
    def mean(self) -> float:
        """均值"""
        mean = 0
        for x, p1, p2 in zip(self.ecdf[0], [0.]+self.ecdf[1], self.ecdf[1]+[1.]):
            mean += x * (p2 - p1)
        return mean

    @property
    def mean_std(self) -> Tuple[float, float]:
        """均值和方差"""
        mean = self.mean
        std = 0
        for x, p1, p2 in zip(self.ecdf[0], [0.]+self.ecdf[1], self.ecdf[1]+[1.]):
            mean += (x - mean) ** 2 * (p2 - p1)
        return mean, std

    def to_json(self) -> dict:
        """返回可以被序列化的json对象"""
        output = super().to_json()
        output['ecdf'] = self.ecdf
        output['num_samples'] = self.num_samples
        return output

    @classmethod
    def from_array(cls, array, bins, max_error_rate, name: str, preds: List[str] = None, succs: List[str] = None):
        """
        从序列中构造数值型节点
        :param array: 序列
        :param bins: 分箱数量
        :param max_error_rate: 最大允许的不能转换成数值的比例
        :param name: 节点名称
        :param preds: 前置节点
        :param succs: 后置节点
        :return:
        """
        import pandas as pd

        miss_rate = pd.isna(array).mean()
        array = pd.to_numeric(array, errors='coerce')
        assert array.isna().mean() - miss_rate <= max_error_rate, '无法转换的数据比例超过{}'.format(max_error_rate)

        x_axis = [i/bins for i in range(1, bins)]
        y_axis = array.quantile(x_axis).tolist()
        # TODO 需要处理y_axis有重复的情况
        return cls(name, ecdf=(x_axis+[array.max()], y_axis), num_samples=array.shape[0], preds=preds, succs=succs)
