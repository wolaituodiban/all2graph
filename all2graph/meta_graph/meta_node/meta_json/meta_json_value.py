import json
from typing import Dict, Union, Type, List

import numpy as np
import pandas as pd

from .meta_array import MetaArray
from .meta_bool import MetaBool
from .meta_null import MetaNull
from .meta_number import MetaNumber
from .meta_object import MetaObject
from .meta_string import MetaString
from .meta_time_stamp import ALL_TIME_UNITS, SECOND_DIFF, MetaTimeStamp
from ..meta_node import MetaNode
from ....stats import Discrete, ECDF
from ....macro import EPSILON, NULL


ALL_TYPES_OF_VALUE: Dict[str, Type[MetaNode]] = {
    'object': MetaObject,
    'array': MetaArray,
    'timestamp': MetaTimeStamp,
    'bool': MetaBool,
    'string': MetaString,
    'number': MetaNumber,
    NULL: MetaNull,
}


class MetaJsonValue(MetaNode):
    __doc__ = """
    参照https://www.graph.org/graph-en.html的标准编制

    json是由collection of name/value pairs和ordered list of values组成的。一般来说list本身会对应一个name。
    为了让所有的value都能对应上一个name，我们把list中的所有value都对应到list本身的name上时，这样所有的value就都有了name。

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

    def __init__(self, meta_data: Dict[str, MetaNode], **kwargs):
        for k, v in meta_data.items():
            assert type(v) == ALL_TYPES_OF_VALUE[k], '{}'.format(meta_data)
        super().__init__(
            meta_data=meta_data, **kwargs
        )

    def __getitem__(self, item):
        return self.meta_data[item]

    def to_discrete(self, **kwargs) -> Discrete:
        return Discrete.from_ecdfs({k: v.freq for k, v in self.meta_data.items()}, **kwargs)

    def to_json(self) -> dict:
        output = super().to_json()
        output['meta_data'] = {k: v.to_json() for k, v in self.meta_data.items()}
        return output

    @classmethod
    def from_json(cls, obj: Union[str, dict]):
        if isinstance(obj, str):
            obj = json.loads(obj)
        else:
            obj = dict(obj)
        obj['meta_data'] = {k: ALL_TYPES_OF_VALUE[k].from_json(v) for k, v in obj['meta_data'].items()}
        return super().from_json(obj)

    @classmethod
    def from_data(cls, num_samples, sample_ids, values, sample_times=None, **kwargs):
        meta_data = {}
        sub_sample_ids = pd.Series(sample_ids)
        sub_values = pd.Series(values)
        # 处理object
        dict_mask = [isinstance(v, dict) for v in sub_values]
        if any(dict_mask):
            temp_sample_ids = sub_sample_ids[dict_mask]
            meta_data['object'] = MetaObject.from_data(
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
            meta_data['array'] = MetaArray.from_data(
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
            meta_data['bool'] = MetaBool.from_data(
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
            meta_data[NULL] = MetaNull.from_data(
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
            meta_data['number'] = MetaNumber.from_data(
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

            meta_data['timestamp'] = MetaTimeStamp.from_data(
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
            meta_data['string'] = MetaString.from_data(
                num_samples=num_samples, sample_ids=sub_sample_ids, values=sub_values, **kwargs
            )
        return super().from_data(
            num_samples=num_samples, sample_ids=sample_ids, values=values, meta_data=meta_data, **kwargs
        )

    @classmethod
    def reduce(cls, structs: List[MetaNode], weights=None, **kwargs):
        if weights is None:
            weights = np.full(len(structs), 1 / len(structs))
        else:
            weights = np.array(weights) / sum(weights)

        temp_data = {}
        for weight, struct in zip(weights, structs):
            for k, v in struct.meta_data.items():
                if k not in temp_data:
                    temp_data[k] = ([v], [weight])
                else:
                    temp_data[k][0].append(v)
                    temp_data[k][1].append(weight)

        meta_data = {k: ALL_TYPES_OF_VALUE[k].reduce(v, weights=w, **kwargs) for k, (v, w) in temp_data.items()}

        for k in temp_data:
            w_sum = sum(temp_data[k][1])
            if w_sum < 1:
                meta_data[k].freq = ECDF.reduce(
                    [meta_data[k].freq, ECDF([0], [1], initialized=True)],
                    weights=[w_sum, 1-w_sum], **kwargs
                )

        return super().reduce(structs, meta_data=meta_data, weights=weights, **kwargs)
