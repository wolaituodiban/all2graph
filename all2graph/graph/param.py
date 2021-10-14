import dgl
import torch

from .raw import RawGraph
from ..version import __version__
from ..preserves import VALUE


class ParamGraph:
    def __init__(self, graph: RawGraph, value: list, mapper: dict):
        self.version = __version__
        self.graph: dgl.DGLGraph = dgl.graph((graph.src, graph.dst), num_nodes=graph.num_nodes)
        self.graph.ndata[VALUE] = torch.tensor(value, dtype=torch.long)
        self.mapper = mapper
        self.embedding = None

    @property
    def value(self):
        return self.graph.ndata[VALUE]

    def to(self, *args, **kwargs):
        self.graph = self.graph.to(*args, **kwargs)
        if self.embedding is not None:
            self.embedding = self.embedding.to(*args, **kwargs)
        return self

    def set_embedding(self, emb):
        self.embedding = emb

    def get_embedding(self, names: list):
        if self.embedding is None:
            return None
        else:
            return self.embedding[[self.mapper[x] for x in names]]

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, ', '.join(self.mapper))
