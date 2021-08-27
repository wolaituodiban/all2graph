from typing import Union, List

from ...json_operator import JsonOperator


class JsonPathNode(JsonOperator):
    def __init__(self):
        super().__init__()
        self.childs_or_processors: List[Union[JsonPathNode, JsonOperator]] = []

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
        if len(self.childs_or_processors) == 0:
            output = self.get(obj)
            if len(output) == 1:
                output = output[0]
            return output
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

    @staticmethod
    def extra_repr():
        return 'root:$'

    def __repr__(self):
        output = self.extra_repr()
        for child_or_processor in self.childs_or_processors:
            strs = str(child_or_processor).split('\n')
            strs[0] = '|____' + strs[0]
            output += ''.join(['\n     ' + line for line in strs])
        return output
