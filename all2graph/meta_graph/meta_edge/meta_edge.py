import json
from typing import Dict, Type

import numpy as np
import pandas as pd

from ..meta_node import MetaNode, ALL_NODE_CLASSES
from ...meta_struct import MetaStruct
from ...stats import ECDF


class MetaEdge(MetaStruct):
    FREQ = 'freq'
    SUCC = 'succ'
    """边基类"""
    def __init__(self, freq: ECDF, succ: MetaNode = None, **kwargs):
        """

        :param succ: 后置节点的元数据
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.freq = freq
        self.succ = succ

    def __eq__(self, other):
        return super().__eq__(other) and self.freq == other.freq and self.succ == other.succ

    def to_json(self) -> dict:
        output = super().to_json()
        output[self.FREQ] = self.freq.to_json()
        if self.succ is not None:
            output[self.SUCC] = self.succ.to_json()
        return output

    @classmethod
    def from_json(cls, obj, classes: Dict[str, Type[MetaNode]] = None):
        if isinstance(obj, str):
            obj = json.loads(obj)
        else:
            obj = dict(obj)
        obj[cls.FREQ] = ECDF.from_json(obj[cls.FREQ])
        if cls.SUCC in obj:
            succ_class = obj[cls.SUCC]['type']
            if classes is not None and succ_class in classes:
                obj[cls.SUCC] = classes[succ_class].from_json(obj[cls.SUCC])
            else:
                obj[cls.SUCC] = ALL_NODE_CLASSES[succ_class].from_json(obj[cls.SUCC])
        return super().from_json(obj)

    @classmethod
    def from_data(cls, num_samples, sample_ids, **kwargs):
        # todo 没有考虑succ
        counts = pd.value_counts(sample_ids)
        if len(counts) < num_samples:
            old_counts = counts
            counts = np.zeros(num_samples)
            counts[:old_counts.shape[0]] = old_counts
        freq = ECDF.from_data(counts, **kwargs)
        kwargs[cls.FREQ] = freq
        return super().from_data(**kwargs)

    @classmethod
    def reduce(cls, structs, **kwargs):
        # todo 没有考虑succ
        freq = ECDF.reduce([struct.freq for struct in structs], **kwargs)
        kwargs[cls.FREQ] = freq
        return super().reduce(structs, **kwargs)
