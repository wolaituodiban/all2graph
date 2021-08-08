import json
from typing import Dict, Type

from ..macro import TYPE
from ..node import MetaNode, ALL_NODE_CLASSES
from ..stats import ECDF


class MetaEdge(ECDF):
    SUCC = 'succ'
    """边基类"""
    def __init__(self, succ: MetaNode = None, **kwargs):
        """

        :param succ: 后置节点的元数据
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.succ = succ

    def to_json(self) -> dict:
        output = super().to_json()
        if self.succ is not None:
            output[self.SUCC] = self.succ.to_json()
        return output

    @classmethod
    def from_json(cls, obj, classes: Dict[str, Type[MetaNode]] = None):
        if isinstance(obj, str):
            obj = json.loads(obj)
        else:
            obj = dict(obj)

        if cls.SUCC in obj:
            succ_class = obj[cls.SUCC][TYPE]
            if classes is not None and succ_class in classes:
                obj[cls.SUCC] = classes[succ_class].from_json(obj[cls.SUCC])
            else:
                obj[cls.SUCC] = ALL_NODE_CLASSES[succ_class].from_json(obj[cls.SUCC])
        return super().from_json(obj)
