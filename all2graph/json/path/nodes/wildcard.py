from .base import JsonPathNode


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
