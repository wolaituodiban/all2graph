import json
from datetime import datetime
import string
from typing import Iterable, Union, List, Tuple, Dict

import pandas as pd

from .json_node_processors import JsonNodeProcessor


class JsonPathNode(JsonNodeProcessor):
    def __init__(self):
        super().__init__()
        self.childs_or_processors: List[Union[JsonPathNode, JsonNodeProcessor]] = []

    def get(self, obj) -> list:
        raise NotImplementedError

    def update(self, obj, processed_objs: list):
        raise NotImplementedError

    def __call__(self, obj, **kwargs):
        """
        计算逻辑，深层优先，同层按照注册顺序排序
        :param obj: 输入的对象
        :param kwargs:
        :return:
        """
        processed_objs = []
        for child_obj in self.get(obj):
            for child_or_processor in self.childs_or_processors:
                child_obj = child_or_processor(child_obj, **kwargs)
            processed_objs.append(child_obj)
        return self.update(obj, processed_objs=processed_objs)

    def is_duplicate(self, other) -> bool:
        raise NotImplementedError

    @classmethod
    def from_json_path(cls, json_path: str):
        """

        :param json_path:
        :return: (对象, 剩余的json_path)
        如果json_path不合法，那么raise AssertionError
        """
        raise NotImplementedError

    def index_of_last_node_processor(self) -> int:
        i = 0
        while i < len(self.childs_or_processors) and not isinstance(self.childs_or_processors[i], JsonPathNode):
            i += 1
        if i == len(self.childs_or_processors):
            i = 0
        return i

    def insert(self, other):
        index_of_last_node_processor = self.index_of_last_node_processor()
        for child_or_processor in self.childs_or_processors[index_of_last_node_processor:]:
            if isinstance(child_or_processor, JsonPathNode) and child_or_processor.is_duplicate(other):
                return child_or_processor
        self.childs_or_processors.append(other)
        return other

    def extra_repr(self):
        return 'root:$'

    def __repr__(self):
        output = self.extra_repr()
        for child_or_processor in self.childs_or_processors:
            strs = str(child_or_processor).split('\n')
            strs[0] = '|____' + strs[0]
            output += ''.join(['\n     ' + line for line in strs])
        return output


class ChildOperator(JsonPathNode):
    def __init__(self, child_name):
        super().__init__()
        self.child_name = child_name

    def get(self, obj) -> list:
        if isinstance(obj, dict) and self.child_name in obj:
            return [obj[self.child_name]]
        else:
            return []

    def update(self, obj, processed_objs: list):
        if len(processed_objs) > 0:
            obj[self.child_name] = processed_objs[0]
        return obj

    def is_duplicate(self, other) -> bool:
        return type(self) == type(other) and self.child_name == other.child_name

    @classmethod
    def from_json_path(cls, json_path: str):
        assert len(json_path) > 2, json_path
        if json_path[1] == '.':
            assert json_path[2] in string.ascii_letters, json_path
            second_dot_pos = json_path[2:].find('.')
            if second_dot_pos < 0:
                second_dot_pos = len(json_path)
            else:
                second_dot_pos += 2
            child_name = json_path[2:second_dot_pos]
            rest_json_path = '$' + json_path[second_dot_pos:]
        elif json_path[1] == '[':
            assert json_path[2] in ('"', "'"), json_path
            right_square_bracket_pos = json_path.find(']')
            assert right_square_bracket_pos > 0, json_path
            child_name = json_path[3:right_square_bracket_pos-1]
            rest_json_path = '$' + json_path[right_square_bracket_pos+1:]
        else:
            raise AssertionError(json_path)
        return cls(child_name=child_name), rest_json_path

    def extra_repr(self):
        return ".{}".format(self.child_name)


class RecursiveDescent(JsonPathNode):
    def __init__(self):
        super().__init__()

    def get(self, obj) -> list:
        output = [obj]
        if isinstance(obj, list):
            for child_obj in obj:
                output += self.get(child_obj)
        elif isinstance(obj, dict):
            for child_obj in obj.values():
                output += self.get(child_obj)
        return output

    def update(self, obj, processed_objs: list):
        return obj

    def is_duplicate(self, other) -> bool:
        return type(self) == type(other)

    @classmethod
    def from_json_path(cls, json_path: str):
        assert json_path.startswith('$..'), json_path
        first_not_dot_pos = 1
        while first_not_dot_pos < len(json_path) and json_path[first_not_dot_pos] == '.':
            first_not_dot_pos += 1
        return cls(), '$'+json_path[first_not_dot_pos:]

    def extra_repr(self):
        return ".."


class Wildcard(JsonPathNode):
    def __init__(self):
        super().__init__()
        self.key_cache = None

    def get(self, obj) -> list:
        if isinstance(obj, list):
            return obj
        elif isinstance(obj, dict):
            self.key_cache = obj.keys()
            return list(obj.values())
        else:
            return []

    def update(self, obj, processed_objs):
        if isinstance(obj, dict):
            for key, value in zip(self.key_cache, processed_objs):
                obj[key] = value
        elif isinstance(obj, list):
            for i, value in enumerate(processed_objs):
                obj[i] = value
        return obj

    def is_duplicate(self, other) -> bool:
        return type(self) == type(other)

    @classmethod
    def from_json_path(cls, json_path: str):
        if json_path.startswith('$.*'):
            return cls(), '$'+json_path[3:]
        elif json_path.startswith('$[*]'):
            return cls(), '$'+json_path[4:]
        else:
            raise AssertionError(json_path)

    def extra_repr(self):
        return ".*"


class SubscriptOperator(JsonPathNode):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def get(self, obj) -> list:
        if isinstance(obj, list) and len(obj) > self.index:
            return [obj[self.index]]
        else:
            return []

    def update(self, obj, processed_objs: list):
        if len(processed_objs) > 0:
            obj[self.index] = processed_objs[0]
        return obj

    def is_duplicate(self, other) -> bool:
        return type(self) == type(other) and self.index == other.index

    @classmethod
    def from_json_path(cls, json_path: str):
        assert len(json_path) > 2 and json_path[1] == '[', json_path
        right_square_bracket_pos = json_path.find(']')
        assert right_square_bracket_pos > 0, json_path
        index = json_path[2:right_square_bracket_pos]
        assert index.isdigit(), json_path
        return cls(index=index), '$'+json_path[right_square_bracket_pos+1:]

    def extra_repr(self):
        return "[{}]".format(self.index)


ALL_JSON_PATH_NODE_CLASSES: Dict[str, JsonPathNode] = {
    k: v for k, v in globals().items() if isinstance(v, type) and issubclass(v, JsonPathNode) and v != JsonPathNode
}


def insert_node(root_node: JsonPathNode, json_path: str, processor: JsonNodeProcessor):
    if json_path in ('$', '$.'):
        root_node.childs_or_processors.append(processor)
    else:
        for cls in ALL_JSON_PATH_NODE_CLASSES.values():
            try:
                node, rest_json_path = cls.from_json_path(json_path)
                node = root_node.insert(node)
                insert_node(node, rest_json_path, processor)
                break
            except AssertionError:
                continue
        else:
            raise ValueError(json_path)


class JsonPathTree(JsonPathNode):
    """
    https://jsonpath.com/
    https://goessner.net/articles/JsonPath/
    """

    def __init__(self, json_col, sample_time_col=None, sample_time_format=None,
                 processors: List[Tuple[str, JsonNodeProcessor]] = None):
        super().__init__()
        self.json_col = json_col
        self.sample_time_col = sample_time_col
        self.sample_time_format = sample_time_format
        if processors is not None:
            for json_path, processor in processors:
                insert_node(self, json_path, processor)

    def get(self, obj) -> list:
        return [obj]

    def update(self, obj, processed_objs: list):
        return processed_objs[0]

    def __call__(self, df: pd.DataFrame, **kwargs) -> Iterable:
        if len(self.childs_or_processors) == 0:
            for value in df[self.json_col]:
                yield json.loads(value)
        elif self.sample_time_col is None:
            for obj in df[self.json_col]:
                try:
                    obj = json.loads(obj)
                    yield super().__call__(obj)
                except json.JSONDecodeError:
                    yield None
        else:
            df = df[[self.json_col, self.sample_time_col]].copy()
            for row in df.itertuples():
                try:
                    obj = json.loads(row[1])
                    sample_time = datetime.strptime(row[2], self.sample_time_format)
                    yield super().__call__(obj, sample_time=sample_time)
                except json.JSONDecodeError:
                    yield None

    def is_duplicate(self, other) -> bool:
        raise NotImplementedError

    @classmethod
    def from_json_path(cls, json_path: str):
        raise NotImplementedError
