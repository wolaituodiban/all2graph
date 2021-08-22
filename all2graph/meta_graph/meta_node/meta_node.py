import json
from typing import Union

import numpy as np
import pandas as pd

from ...meta_struct import MetaStruct
from ...stats import ECDF


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
    def __init__(self, count: ECDF, meta_data=None, **kwargs):
        """

        :param count: 节点的计数分布
        :param meta_data: 节点的元数据
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.count = count
        self.meta_data = meta_data

    def __eq__(self, other):
        return super().__eq__(other) and self.count == other.count_ecdf and self.meta_data == other.value_ecdf

    def to_json(self) -> dict:
        """将对象装化成可以被json序列化的对象"""
        output = super().to_json()
        output['count'] = self.count.to_json()
        output['meta_data'] = self.meta_data
        return output

    @classmethod
    def from_json(cls, obj: Union[str, dict]):
        if isinstance(obj, str):
            obj = json.loads(obj)
        else:
            obj = dict(obj)
        obj['count'] = ECDF.from_json(obj['count'])
        return super().from_json(obj)

    @classmethod
    def from_data(cls, num_samples: int, sample_ids, values, **kwargs):
        """根据向量生成元节点"""
        node_counts = pd.value_counts(sample_ids).values
        if node_counts.shape[0] < num_samples:
            old_node_counts = node_counts
            node_counts = np.zeros(num_samples)
            node_counts[:old_node_counts.shape[0]] = old_node_counts
        else:
            assert node_counts.shape[0] == num_samples
        return super().from_data(count=ECDF.from_data(node_counts, **kwargs), **kwargs)

    @classmethod
    def reduce(cls, structs, **kwargs):
        """
        合并多个经验累计分布函数，返回一个贾总的经验累计分布函数
        会自动解析update_records，并生成一个合并后的update_records
        """
        freq = ECDF.reduce([struct.count_ecdf for struct in structs], **kwargs)
        return super().reduce(structs, count=freq, **kwargs)
