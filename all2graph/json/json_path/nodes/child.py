import string
from .base import JsonPathNode


class Child(JsonPathNode):
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
