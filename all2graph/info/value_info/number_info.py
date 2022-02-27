import numpy as np

from ...meta_struct import MetaStruct
from ...stats import ECDF


class NumberInfo(MetaStruct):
    def __init__(self, count: ECDF, value: ECDF, **kwargs):
        """

        :param.py count_ecdf: 节点的计数分布
        :param.py value_ecdf: 节点的元数据
        :param.py kwargs:
        """
        super().__init__(**kwargs)
        self.count = count
        self.value = value

    def __eq__(self, other):
        return super().__eq__(other) \
               and self.count == other.count \
               and self.value == other.value

    def to_json(self) -> dict:
        output = super().to_json()
        output['count'] = self.count.to_json()
        output['value'] = self.value.to_json()
        return output

    @classmethod
    def from_json(cls, obj: dict):
        obj = dict(obj)
        obj['count'] = ECDF.from_json(obj['count'])
        obj['value'] = ECDF.from_json(obj['value'])
        return super().from_json(obj)

    @classmethod
    def from_data(cls, counts, values, num_bins=None):
        count = ECDF.from_data(counts, num_bins=num_bins)
        value = ECDF.from_data(values, num_bins=num_bins)
        return super().from_data(count=count, value=value)

    @classmethod
    def reduce(cls, structs, weights=None, num_bins=None):
        count = ECDF.reduce([struct.count for struct in structs], weights=weights, num_bins=num_bins)
        # info data的weight可以从freq中推出
        value = ECDF.reduce(
            [struct.value for struct in structs],
            weights=[w * struct.count.mean for w, struct in zip(weights, structs)],
            num_bins=num_bins)
        return super().reduce(structs, count=count, value=value, weights=weights)

    def extra_repr(self) -> str:
        return 'count={}\nvalue={}'.format(self.count, self.value)

    def get_probs(
            self, q, bounds_error=False, fill_value=(0, 1), assume_sorted=True, **kwargs
    ) -> np.ndarray:
        """

        :param q: 分位数
        :param bounds_error:
        :param fill_value:
        :param assume_sorted:
        :param kwargs:
        :return: 分位数对应的累计概率
        """
        return self.value.get_probs(
            q=q, bounds_error=bounds_error, fill_value=fill_value, assume_sorted=assume_sorted, **kwargs
        )

    def get_quantiles(self, p, bounds_error=False, fill_value="extrapolate", assume_sorted=True,
                      **kwargs) -> np.ndarray:
        """

        :param p: 累积概率
        :param bounds_error:
        :param assume_sorted:
        :param fill_value:
        :param kwargs:
        :return: 累积概率对应的分位数
        """
        return self.value.get_quantiles(
            p=p, bounds_error=bounds_error, fill_value=fill_value, assume_sorted=assume_sorted, **kwargs)

    def minmax_scale(self, *args, **kwargs):
        """
        Args:
            详情见ECDF.minmax_scale

        Returns:

        """
        return self.value.minmax_scale(*args, **kwargs)
