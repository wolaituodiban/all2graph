from .graph_dataset import GraphDataset
from ..data_parser import DataParser
from ...graph.graph_transer import GraphTranser


class RealtimeDataset(GraphDataset):
    def __init__(self, paths, preprocessor, data_parser, graph_transer, partitions=1, shuffle=False, disable=True):
        pass