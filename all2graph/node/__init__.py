from .category import Category
from .meta_node import MetaNode
from .number import Number


ALL_NODE_CLASSES = {k: v for k, v in globals().items() if isinstance(v, type) and issubclass(v, MetaNode)}