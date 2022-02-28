import dgl

from ..globals import KEY, VALUE, TOKEN, NUMBER, SID, TARGET, KEY2TARGET
from ..meta_struct import MetaStruct


class Graph(MetaStruct):
    def __init__(self, graph: dgl.DGLHeteroGraph, **kwargs):
        super().__init__(initialized=True, **kwargs)
        self.graph = graph

    @classmethod
    def from_data(cls, edges, key_tokens, sids, value_tokens, numbers, **kwargs):
        graph = dgl.heterograph(
            edges,
            num_nodes_dict={KEY: key_tokens.shape[0], VALUE: value_tokens.shape[0], TARGET: len(edges[KEY2TARGET][0])}
        )
        graph.ndata[KEY][TOKEN] = key_tokens
        graph.ndata[VALUE][SID] = sids
        graph.ndata[VALUE][TOKEN] = value_tokens
        graph.ndata[VALUE][NUMBER] = numbers
        return cls(graph, **kwargs)

    def __eq__(self, other, debug=False):
        raise NotImplementedError

    def __repr__(self):
        return self.graph.__repr__()

    @classmethod
    def from_json(cls, obj: dict):
        raise NotImplementedError

    def to_json(self) -> dict:
        raise NotImplementedError

    def to(self, *args, **kwargs):
        self.graph = self.graph.to(*args, **kwargs)
        return self

    def pin_memory(self):
        for k, v in self.graph.ndata.items():
            for kk, vv in v.items():
                self.graph.ndata[k][kk] = vv.pin_memory()
        for k, v in self.graph.edata.items():
            for kk, vv in v.items():
                self.graph.edata[k][kk] = vv.pin_memory()
        return self

    def component_subgraph(self, i):
        raise NotImplementedError

    @classmethod
    def batch(cls, graphs):
        return cls(dgl.batch([graph.graph for graph in graphs]))

    @classmethod
    def reduce(cls, **kwargs):
        raise NotImplementedError
