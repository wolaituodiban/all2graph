from .string_node import StringNode
from .meta_node import MetaNode
from .timestamp import TimeStamp


ALL_NODE_CLASSES = {k: v for k, v in globals().items() if isinstance(v, type) and issubclass(v, MetaNode)}
