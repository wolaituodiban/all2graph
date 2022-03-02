import dgl
import torch

from ..globals import KEY, VALUE, TOKEN, NUMBER, SID, EDGE
from ..meta_struct import MetaStruct


class Graph(MetaStruct):
    def __init__(self, graph: dgl.DGLHeteroGraph, **kwargs):
        super().__init__(initialized=True, **kwargs)
        self.graph = graph

    @classmethod
    def from_data(cls, edges, key_tokens, sids, value_tokens, numbers, **kwargs):
        num_nodes_dict = {KEY: key_tokens.shape[0], VALUE: value_tokens.shape[0]}
        for (_, _, vtype), (u, v) in edges.items():
            if vtype in num_nodes_dict:
                continue
            num_nodes_dict[vtype] = len(v)
        graph = dgl.heterograph(edges, num_nodes_dict=num_nodes_dict)
        graph.nodes[KEY].data[TOKEN] = key_tokens
        graph.nodes[VALUE].data[SID] = sids
        graph.nodes[VALUE].data[TOKEN] = value_tokens
        graph.nodes[VALUE].data[NUMBER] = numbers
        return cls(graph, **kwargs)

    def __eq__(self, other, debug=False):
        raise NotImplementedError

    def __repr__(self):
        return self.graph.__repr__()

    @property
    def key_token(self):
        return self.graph.nodes[KEY].data[TOKEN]

    @property
    def value_token(self):
        return self.graph.nodes[VALUE].data[TOKEN]

    @property
    def number(self):
        return self.graph.nodes[VALUE].data[NUMBER]

    @property
    def key_graph(self) -> dgl.DGLHeteroGraph:
        return dgl.node_type_subgraph(self.graph, [KEY])

    @property
    def value_graph(self) -> dgl.DGLHeteroGraph:
        return dgl.node_type_subgraph(self.graph, [VALUE])

    @property
    def readout_types(self):
        return {ntype for ntype in self.graph.ntypes if ntype != KEY and ntype != VALUE}

    def push_feats(self, feats: torch.Tensor, utype: str, vtype: str, sort=False):
        """将key的feature传播到"""
        u, v = self.graph.edges(etype=(utype, EDGE, vtype))
        if sort:
            raise NotImplementedError
        else:
            return feats[u]

    @classmethod
    def from_json(cls, obj: dict):
        raise NotImplementedError

    def to_json(self) -> dict:
        raise NotImplementedError

    @classmethod
    def load(cls, path, **kwargs):
        graphs, labels = dgl.load_graphs(path)
        return cls(graph=graphs[0]), labels

    def save(self, path, **kwargs):
        dgl.save_graphs(path, [self.graph], **kwargs)

    def to(self, *args, **kwargs):
        self.graph = self.graph.to(*args, **kwargs)
        return self

    def pin_memory(self):
        for ntype in self.graph.ntypes:
            for k, v in self.graph.nodes[ntype].data.items():
                self.graph.nodes[ntype].data[k] = v.pin_memory()
        for etype in self.graph.canonical_etypes:
            for k, v in self.graph.edges[etype].data.items():
                self.graph.nodes[etype].data[k] = v.pin_memory()
        return self

    def component_subgraph(self, i):
        raise NotImplementedError

    @classmethod
    def batch(cls, graphs):
        return cls(dgl.batch([graph.graph for graph in graphs]))

    @classmethod
    def reduce(cls, **kwargs):
        raise NotImplementedError
