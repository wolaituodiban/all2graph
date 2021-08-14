from .meta_edge import MetaEdge


ALL_EDGE_CLASSES = {k: v for k, v in globals().items() if isinstance(v, type) and issubclass(v, MetaEdge)}
