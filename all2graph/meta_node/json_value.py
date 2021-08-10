import json
from typing import Dict, Union, Type

import numpy as np
import pandas as pd

from .array_node import ArrayNode
from .bool_node import BoolNode
from .meta_node import MetaNode
from .null_node import NullNode
from .number import Number
from .object_node import ObjectNode
from .string_node import StringNode
from .timestamp import ALL_TIME_UNITS, SECOND_DIFF, TimeStamp
from ..macro import TYPE
from ..stats import ECDF


ALL_TYPES_OF_VALUE: Dict[str, Type[MetaNode]] = {
    'object': ObjectNode,
    'array': ArrayNode,
    'timestamp': TimeStamp,
    'bool': BoolNode,
    'string': StringNode,
    'number': Number,
    'null': NullNode,
}


class JsonValue(MetaNode):
    __doc__ = """
    参照https://www.json.org/json-en.html的标准编制

    json是由collection of name/value pairs和ordered list of values组成的。一般来说list本身会对应一个name。
    为了让所有的value都能对应上一个name，我们把list中的所有value都对应到list本身的name上时，这样所有的value就都有了name上。

    JsonValue对象会记录某一个name的各种不同value的出现频率和分布
    
    JsonValue目前拓展了json的value的类型，包括一下几种：
        string：一般的value会被归为离散值
        number：一切pd.to_numeric能转换的对象
        object: collection of name/value pairs，一切isinstance(obj, dict)的对象
        array：ordered list of values，一切isinstance(obj, list)的对象
        true：对应True
        false：对应False
        null：一切pd.isna为True的对象
        timestamp：特殊的字符串，一切pd.to_datetime能转换的对象
        
    为了保证完备性，每一个value会按照上述所述的类型顺序，被唯一的归入其中一类。
    """.format(SECOND_DIFF, ALL_TIME_UNITS)

    def __init__(self, value_dist: Dict[str, MetaNode], **kwargs):
        for k, v in value_dist.items():
            assert type(v) == ALL_TYPES_OF_VALUE[k]
        super().__init__(
            node_freq=ECDF.merge(node.node_freq for node in value_dist.values()),
            value_dist=value_dist, **kwargs
        )

    def to_json(self) -> dict:
        output = super().to_json()
        output[self.VALUE_DIST] = {k: v.to_json() for k, v in self.value_dist.items()}
        return output

    @classmethod
    def from_json(cls, obj: Union[str, dict]):
        if isinstance(obj, str):
            obj = json.loads(obj)
        else:
            obj = dict(obj)
        obj[cls.VALUE_DIST] = {k: ALL_TYPES_OF_VALUE[v[TYPE]] for k, v in obj[cls.VALUE_DIST].items()}
        return super().from_json(obj)

    @classmethod
    def from_data(cls, num_samples, sample_ids, values, sample_times=None, **kwargs):
        value_dist = {}
        # 处理时间戳
        timestamps = pd.to_datetime(values, errors='coerce')
        time_mask = pd.notna(timestamps)
        if time_mask.any():
            if sample_times is not None:
                sample_times = sample_times[time_mask]
            temp_sample_ids = sample_ids[time_mask]

            value_dist['timestamp'] = TimeStamp.from_data(
                num_samples=temp_sample_ids.shape[0], sample_ids=temp_sample_ids, values=timestamps[time_mask],
                sample_times=sample_times,
                **kwargs
            )
            mask = np.bitwise_not(time_mask)
            sample_ids = sample_ids[mask]
            values = values[mask]
            num_samples -= temp_sample_ids.shape[0]
        del time_mask

        # 处理null
        null_mask = pd.isna(values)
        if null_mask.any():
            temp_sample_ids = sample_ids[null_mask]
            value_dist['null'] = NullNode.from_data(
                num_samples=temp_sample_ids.shape[0], sample_ids=temp_sample_ids, values=None, **kwargs
            )
            mask = np.bitwise_not(null_mask)
            sample_ids = sample_ids[mask]
            values = values[mask]
            num_samples -= temp_sample_ids.shape[0]
        del null_mask

        # 处理object
        dict_mask = [isinstance(v, dict) for v in values]
        if any(dict_mask):
            temp_sample_ids = sample_ids[dict_mask]
            num_samples = temp_sample_ids.shape[0]
            value_dist['object'] = ObjectNode.from_data(
                num_samples=temp_sample_ids.shape[0], sample_ids=temp_sample_ids, values=None, **kwargs
            )
            mask = np.bitwise_not(dict_mask)
            sample_ids = sample_ids[mask]
            values = values[mask]
            num_samples -= temp_sample_ids.shape[0]
        del dict_mask

        # 处理array
        array_mask = [isinstance(v, list) for v in values]
        if any(array_mask):
            temp_sample_ids = sample_ids[array_mask]
            num_samples = temp_sample_ids.shape[0]
            value_dist['object'] = ArrayNode.from_data(
                num_samples=temp_sample_ids.shape[0], sample_ids=temp_sample_ids, values=None, **kwargs
            )
            mask = np.bitwise_not(array_mask)
            sample_ids = sample_ids[mask]
            values = values[mask]
            num_samples -= temp_sample_ids.shape[0]
        del array_mask

        # 处理


