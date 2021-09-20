from .base import JsonPathNode


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
