try:
    from .graph import Graph
except ImportError:
    Graph = None
from .raw_graph import RawGraph
