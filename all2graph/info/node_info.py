from typing import List, Union

from .str_info import StrInfo
from ..meta_struct import MetaStruct
from ..stats import ECDF
from ..globals import *


class NodeInfo(MetaStruct):
    def __init__(
            self, num_samples: int, num_nodes_ecdf: ECDF, num_pos_samples: int, num_frac_ecdf: ECDF,
            num_count: int, num_ecdf: Union[ECDF, None], str_info: Union[StrInfo, None], **kwargs):
        """
        一类节点的数值统计信息
        Args:
            num_nodes_ecdf: 节点数量分布
            num_pos_samples: 包含此类节点的样本数量
            num_frac_ecdf: 数值型占比分布
            num_count: 数值型节点数量
            num_ecdf: 数值分布
            str_info: 字符串统计信息
            **kwargs:
        """
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.num_nodes_ecdf = num_nodes_ecdf
        self.num_pos_samples = num_pos_samples  # 点数量大于0的样本的数量
        self.num_frac_ecdf = num_frac_ecdf
        self.num_count = num_count
        self.num_ecdf = num_ecdf
        self.str_info = str_info

    @classmethod
    def empty(cls, num_samples):
        return cls(
            num_samples, num_nodes_ecdf=ECDF.from_data([0]),
            num_pos_samples=0, num_frac_ecdf=ECDF.from_data([0]),
            num_count=0, num_ecdf=None, str_info=None, initialized=True)

    @classmethod
    def from_data(cls, num_samples, df, num_bins=None, fast=True, **kwargs):
        num_count = df[NUMBER].count()
        if num_count > 0:
            num_ecdf = ECDF.from_data(df[NUMBER])
        else:
            num_ecdf = None

        if fast:
            num_nodes_ecdf = None
            num_pos_samples = None
            num_frac_ecdf = None
            str_info = None
        else:
            count_df = df.groupby(SAMPLE).agg({SAMPLE: 'count', NUMBER: 'count'})
            num_pos_samples = count_df.shape[0]
            num_nodes = count_df[SAMPLE].tolist() + [0] * (num_samples - num_pos_samples)
            num_nodes_ecdf = ECDF.from_data(num_nodes, num_bins=num_bins)
            num_frac_ecdf = ECDF.from_data(count_df[NUMBER] / count_df[SAMPLE], num_bins=num_bins)
            str_info = StrInfo.from_data(num_pos_samples, df, num_bins=num_bins)

        return super().from_data(num_samples=num_samples, num_nodes_ecdf=num_nodes_ecdf,
                                 num_pos_samples=num_pos_samples, num_frac_ecdf=num_frac_ecdf,
                                 num_count=num_count, num_ecdf=num_ecdf,
                                 str_info=str_info)

    @classmethod
    def batch(cls, node_infos: List, num_bins=None, fast=True, **kwargs):
        num_sampless = []
        num_nodes_ecdfs = []
        num_pos_sampless = []
        num_frac_ecdfs = []
        num_counts = []
        num_ecdfs = []
        str_infos = []
        for info in node_infos:
            num_sampless.append(info.num_samples)
            num_nodes_ecdfs.append(info.num_nodes_ecdf)
            num_pos_sampless.append(info.num_pos_samples)
            num_frac_ecdfs.append(info.num_frac_ecdf)
            if info.num_ecdf is not None:
                num_counts.append(info.num_count)
                num_ecdfs.append(info.num_ecdf)
            if info.str_info is not None:
                str_infos.append(info.str_info)

        if len(num_ecdfs) > 0:
            num_ecdf = ECDF.batch(num_ecdfs, weights=num_counts, num_bins=num_bins)
            num_count = sum(num_counts)
        else:
            num_count = 0
            num_ecdf = None

        if fast:
            num_pos_samples = None
            num_nodes_ecdf = None
            num_frac_ecdf = None
            str_info = None
        else:
            num_pos_samples = sum(num_sampless)
            num_nodes_ecdf = ECDF.batch(num_nodes_ecdfs, weights=num_sampless, num_bins=num_bins)
            num_frac_ecdf = ECDF.batch(num_frac_ecdfs, weights=num_pos_samples, num_bins=num_bins)
            if len(str_infos) > 0:
                str_info = StrInfo.batch(str_infos, num_bins=num_bins)
            else:
                str_info = None
        return super().from_data(num_samples=sum(num_sampless), num_nodes_ecdf=num_nodes_ecdf,
                                 num_pos_samples=num_pos_samples, num_frac_ecdf=num_frac_ecdf,
                                 num_count=num_count, num_ecdf=num_ecdf,
                                 str_info=str_info)

    @property
    def num_frac(self):
        return self.num_frac_ecdf.mean

    @property
    def num_unique_str(self):
        if self.str_info is None:
            return None
        return self.str_info.num_unique_str

    def extra_repr(self) -> str:
        return 'num_unique_str={}, num_frac={}'.format(
            self.num_unique_str, self.num_frac)
