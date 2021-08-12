import json
from typing import Union

import numpy as np
import pandas as pd

from ..meta_struct import MetaStruct
from ..stats import ECDF


class MetaNode(MetaStruct):
    NODE_FREQ = 'node_freq'
    VALUE_DIST = 'value_dist'
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
    def __init__(self, node_freq: ECDF, value_dist=None, **kwargs):
        """
        :params node_freq: 节点数量的频率分布
        :params value_dist: 节点值的分布
        """
        super().__init__(**kwargs)
        self.node_freq = node_freq
        self.value_dist = value_dist

    def __eq__(self, other):
        return super().__eq__(other) and self.node_freq == other.node_freq and self.value_dist == other.value_dist

    @property
    def num_samples(self) -> int:
        return self.node_freq.num_samples

    @property
    def num_nodes(self) -> int:
        return self.node_freq.mean * self.node_freq.num_samples

    def to_json(self) -> dict:
        """将对象装化成可以被json序列化的对象"""
        output = super().to_json()
        output[self.NODE_FREQ] = self.node_freq.to_json()
        output[self.VALUE_DIST] = self.value_dist
        return output

    @classmethod
    def from_json(cls, obj: Union[str, dict]):
        if isinstance(obj, str):
            obj = json.loads(obj)
        else:
            obj = dict(obj)
        obj[cls.NODE_FREQ] = ECDF.from_json(obj[cls.NODE_FREQ])
        return super().from_json(obj)

    @classmethod
    def from_data(cls, num_samples, sample_ids, values, **kwargs):
        """根据向量生成元节点"""
        node_counts = pd.value_counts(sample_ids).values
        if node_counts.shape[0] < num_samples:
            old_node_counts = node_counts
            node_counts = np.zeros(num_samples)
            node_counts[:old_node_counts.shape[0]] = old_node_counts
        else:
            assert node_counts.shape[0] == num_samples
        kwargs[cls.NODE_FREQ] = ECDF.from_data(node_counts)
        return super().from_data(**kwargs)

    @classmethod
    def reduce(cls, structs, **kwargs):
        """
        合并多个经验累计分布函数，返回一个贾总的经验累计分布函数
        会自动解析update_records，并生成一个合并后的update_records
        """
        node_freqs = [struct.node_freq for struct in structs]
        kwargs[cls.NODE_FREQ] = ECDF.reduce(node_freqs)
        return super().reduce(structs, **kwargs)
