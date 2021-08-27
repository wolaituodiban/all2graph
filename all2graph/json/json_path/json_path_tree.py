import json
from datetime import datetime
from typing import Iterable, Union, List, Tuple

import pandas as pd

from .nodes import JsonPathNode, ALL_JSON_PATH_NODE_CLASSES
from ..json_operator import JsonOperator


def insert_node(root_node: JsonPathNode, json_path: str, processor: JsonOperator = None):
    if json_path in ('$', '$.'):
        if processor is not None:
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
                 processors: List[Union[Tuple[str], Tuple[str, JsonOperator]]] = None):
        super().__init__()
        self.json_col = json_col
        self.sample_time_col = sample_time_col
        self.sample_time_format = sample_time_format
        if processors is not None:
            for args in processors:
                insert_node(self, *args)

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
