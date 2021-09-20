from typing import Dict
from .base import JsonPathNode
from .child import Child
from .recursive_descent import RecursiveDescent
from .subscribe import Subscript
from .wildcard import Wildcard


ALL_JSON_PATH_NODE_CLASSES: Dict[str, JsonPathNode] = {
    k: v for k, v in globals().items() if isinstance(v, type) and issubclass(v, JsonPathNode) and v != JsonPathNode
}
