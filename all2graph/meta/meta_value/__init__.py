from .meta_value import MetaValue
from .meta_number import MetaNumber
from .meta_string import MetaString


ALL_NODE_CLASSES = {k: v for k, v in globals().items() if isinstance(v, type) and issubclass(v, MetaValue)}
