from .meta_node import MetaNode
from .meta_json import *


ALL_NODE_CLASSES = {k: v for k, v in globals().items() if isinstance(v, type) and issubclass(v, MetaNode)}
