from .meta_node import MetaNode
from .meta_number import MetaNumber
from .meta_string import MetaString
from .meta_time_stamp import MetaTimeStamp


ALL_NODE_CLASSES = {k: v for k, v in globals().items() if isinstance(v, type) and issubclass(v, MetaNode)}
