try:
    from .graph import Graph
    from .param import ParamGraph
except ImportError:
    pass
from .raw import RawGraph
