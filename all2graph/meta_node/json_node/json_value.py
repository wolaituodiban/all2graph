import json
from typing import Dict, Union, Type, List

import numpy as np
import pandas as pd

from .array_node import ArrayNode
from .bool_node import BoolNode
from .null_node import NullNode
from .number import Number
from .object_node import ObjectNode
from .string_node import StringNode
from .timestamp import ALL_TIME_UNITS, SECOND_DIFF, TimeStamp
from ..meta_node import MetaNode
from ...stats import Discrete, ECDF
from ...macro import EPSILON


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
    参照https://www.graph.org/graph-en.html的标准编制

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
            assert type(v) == ALL_TYPES_OF_VALUE[k], '{}'.format(value_dist)
        super().__init__(
            value_dist=value_dist, **kwargs
        )
        assert abs(self.num_nodes - sum(dist.num_nodes for dist in self.value_dist.values())) < EPSILON, '{}'.format(
            json.dumps(self.to_json(), indent=2)
        )
        assert set(dist.num_samples for dist in self.value_dist.values()) == {self.num_samples}, '{}'.format(
            {k: dist.num_samples for k, dist in self.value_dist.items()}
        )

    def to_discrete(self, **kwargs) -> Discrete:
        return Discrete.from_ecdfs({k: v.node_freq for k, v in self.value_dist.items()}, **kwargs)

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
        obj[cls.VALUE_DIST] = {k: ALL_TYPES_OF_VALUE[k].from_json(v) for k, v in obj[cls.VALUE_DIST].items()}
        return super().from_json(obj)

    @classmethod
    def from_data(cls, num_samples, sample_ids, values, sample_times=None, **kwargs):
        value_dist = {}
        sub_sample_ids = pd.Series(sample_ids)
        sub_values = pd.Series(values)
        # 处理object
        dict_mask = [isinstance(v, dict) for v in sub_values]
        if any(dict_mask):
            temp_sample_ids = sub_sample_ids[dict_mask]
            value_dist['object'] = ObjectNode.from_data(
                num_samples=num_samples, sample_ids=temp_sample_ids, values=None, **kwargs
            )
            mask = np.bitwise_not(dict_mask)
            sub_sample_ids = sub_sample_ids[mask]
            sub_values = sub_values[mask]
            if sample_times is not None:
                sample_times = pd.Series(sample_times)[mask]
        del dict_mask

        # 处理array
        array_mask = [isinstance(v, list) for v in sub_values]
        if any(array_mask):
            temp_sample_ids = sub_sample_ids[array_mask]
            value_dist['array'] = ArrayNode.from_data(
                num_samples=num_samples, sample_ids=temp_sample_ids, values=None, **kwargs
            )
            mask = np.bitwise_not(array_mask)
            sub_sample_ids = sub_sample_ids[mask]
            sub_values = sub_values[mask]
            if sample_times is not None:
                sample_times = pd.Series(sample_times)[mask]
        del array_mask

        # 处理bool
        bool_mask = [isinstance(v, bool) for v in sub_values]
        if any(bool_mask):
            temp_sample_ids = sub_sample_ids[bool_mask]
            sub_num_samples = temp_sample_ids.shape[0]
            value_dist['bool'] = BoolNode.from_data(
                num_samples=num_samples, sample_ids=temp_sample_ids, values=sub_values[bool_mask], **kwargs
            )
            mask = np.bitwise_not(bool_mask)
            sub_sample_ids = sub_sample_ids[mask]
            sub_values = sub_values[mask]
            sub_num_samples -= temp_sample_ids.shape[0]
            if sample_times is not None:
                sample_times = pd.Series(sample_times)[mask]
        del bool_mask

        # 处理null
        null_mask = pd.isna(sub_values)
        if null_mask.any():
            temp_sample_ids = sub_sample_ids[null_mask]
            value_dist['null'] = NullNode.from_data(
                num_samples=num_samples, sample_ids=temp_sample_ids, values=None, **kwargs
            )
            mask = np.bitwise_not(null_mask)
            sub_sample_ids = sub_sample_ids[mask]
            sub_values = sub_values[mask]
            if sample_times is not None:
                sample_times = pd.Series(sample_times)[mask]
        del null_mask

        # 处理number
        number = pd.to_numeric(sub_values, errors='coerce')
        num_mask = pd.notna(number)
        if num_mask.any():
            temp_sample_ids = sub_sample_ids[num_mask]
            value_dist['number'] = Number.from_data(
                num_samples=num_samples, sample_ids=temp_sample_ids, values=number[num_mask], **kwargs
            )
            mask = np.bitwise_not(num_mask)
            sub_sample_ids = sub_sample_ids[mask]
            sub_values = sub_values[mask]
            if sample_times is not None:
                sample_times = pd.Series(sample_times)[mask]
        del num_mask

        # 处理timestamp
        timestamps = pd.to_datetime(sub_values, errors='coerce')
        time_mask = pd.notna(timestamps)
        if time_mask.any():
            if sample_times is not None:
                sample_times = pd.Series(sample_times)[time_mask]
            temp_sample_ids = sub_sample_ids[time_mask]

            value_dist['timestamp'] = TimeStamp.from_data(
                num_samples=num_samples, sample_ids=temp_sample_ids, values=timestamps[time_mask],
                sample_times=sample_times,
                **kwargs
            )
            mask = np.bitwise_not(time_mask)
            sub_sample_ids = sub_sample_ids[mask]
            sub_values = sub_values[mask]
        del time_mask

        # 处理string
        if sub_sample_ids.shape[0] > 0:
            value_dist['string'] = StringNode.from_data(
                num_samples=num_samples, sample_ids=sub_sample_ids, values=sub_values, **kwargs
            )
        kwargs[cls.VALUE_DIST] = value_dist
        return super().from_data(num_samples=num_samples, sample_ids=sample_ids, values=values, **kwargs)

    @classmethod
    def reduce(cls, structs: List[MetaNode], **kwargs):
        num_samples = 0
        value_dist = {}
        for struct in structs:
            num_samples += struct.num_samples
            for k, v in struct.value_dist.items():
                if k not in value_dist:
                    value_dist[k] = [v]
                else:
                    value_dist[k].append(v)

        value_dist = {k: ALL_TYPES_OF_VALUE[k].reduce(v) for k, v in value_dist.items()}

        for k in value_dist:
            if value_dist[k].num_samples < num_samples:
                value_dist[k].node_freq = ECDF.reduce(
                    [
                        value_dist[k].node_freq,
                        ECDF.from_data(np.zeros(num_samples-value_dist[k].node_freq.num_samples), **kwargs)]
                )

        kwargs[cls.VALUE_DIST] = value_dist
        return super().reduce(structs, **kwargs)
