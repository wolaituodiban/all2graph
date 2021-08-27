from .base import JsonPathNode


class Subscript(JsonPathNode):
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