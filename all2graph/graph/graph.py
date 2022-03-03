import json
import gzip
import pickle

import numpy as np
import dgl
import dgl.function as fn
import torch

from ..globals import KEY, VALUE, TOKEN, NUMBER, SID, KEY2VALUE, EDGE, KEY2KEY, VALUE2VALUE
from ..meta_struct import MetaStruct


def tensor2list(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy().tolist()


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

    def push_key2value(self, feats: torch.Tensor) -> torch.Tensor:
        with self.graph.local_scope():
            self.graph.nodes[KEY].data['feat'] = feats
            self.graph.push(
                torch.arange(self.graph.num_nodes(KEY), device=self.graph.device),
                message_func=fn.copy_u('feat', 'feat'),
                reduce_func=fn.sum('feat', 'feat'),
                etype=KEY2VALUE
            )
            return self.graph.nodes[VALUE].data['feat']

    def push_key2readout(self, feats: torch.Tensor, ntype: str) -> torch.Tensor:
        with self.graph.local_scope():
            self.graph.nodes[KEY].data['feat'] = feats
            self.graph.push(
                torch.arange(self.graph.num_nodes(KEY), device=self.graph.device),
                message_func=fn.copy_u('feat', 'feat'),
                reduce_func=fn.sum('feat', 'feat'),
                etype=(KEY, EDGE, ntype)
            )
            return self.graph.nodes[ntype].data['feat']

    def push_value2readout(self, feats: torch.Tensor, ntype: str) -> torch.Tensor:
        with self.graph.local_scope():
            self.graph.nodes[VALUE].data['feat'] = feats
            self.graph.push(
                torch.arange(self.graph.num_nodes(VALUE), device=self.graph.device),
                message_func=fn.copy_u('feat', 'feat'),
                reduce_func=fn.sum('feat', 'feat'),
                etype=(VALUE, EDGE, ntype)
            )
            return self.graph.nodes[ntype].data['feat']

    @classmethod
    def from_json(cls, obj: dict):
        edges = {}
        key_tokens = torch.tensor(np.array(obj['key_tokens']), dtype=torch.long)
        sids = torch.tensor(np.array(obj['sids']), dtype=torch.long)
        value_tokens = torch.tensor(np.array(obj['value_tokens']), dtype=torch.long)
        numbers = torch.tensor(np.array(obj['numbers']), dtype=torch.float32)
        for etype, (u, v) in obj['edges'].items():
            u = torch.tensor(np.array(u), dtype=torch.long)
            v = torch.tensor(np.array(v), dtype=torch.long)
            edges[tuple(json.loads(etype))] = (u, v)
        return cls.from_data(edges=edges, key_tokens=key_tokens, sids=sids, value_tokens=value_tokens, numbers=numbers)

    def to_json(self) -> dict:
        output = {'edges': {}}
        for etype in self.graph.canonical_etypes:
            u, v = self.graph.edges(etype=etype)
            output['edges'][json.dumps(etype)] = [tensor2list(u), tensor2list(v)]
        output['key_tokens'] = tensor2list(self.graph.nodes[KEY].data[TOKEN])
        output['sids'] = tensor2list(self.graph.nodes[VALUE].data[SID])
        output['value_tokens'] = tensor2list(self.graph.nodes[VALUE].data[TOKEN])
        output['numbers'] = tensor2list(self.graph.nodes[VALUE].data[NUMBER])
        return output

    def to_simple(self, writeback_mapping=False, **kwargs):
        if writeback_mapping:
            graph, wm = dgl.to_simple(self.graph, writeback_mapping=writeback_mapping, **kwargs)
            self.graph = graph
            return Graph(graph), wm
        else:
            graph = dgl.to_simple(self.graph, writeback_mapping=writeback_mapping, **kwargs)
            return Graph(graph)

    def add_self_loop(self, etype=None):
        if etype is None:
            graph = dgl.add_self_loop(self.graph, etype=KEY2KEY)
            graph = dgl.add_self_loop(graph, etype=VALUE2VALUE)
        else:
            graph = dgl.add_self_loop(self.graph, etype)
        return Graph(graph)

    def add_edges_by_key(self, degree, r_degree, keys=None):
        kid, vid = self.graph.edges(etype=KEY2VALUE)
        num_keys = torch.unique(kid).shape[0]

    @classmethod
    def load(cls, path, **kwargs):
        with gzip.open(path, 'rb', **kwargs) as file:
            return pickle.load(file)

    def save(self, path, labels=None, **kwargs):
        with gzip.open(path, 'wb', **kwargs) as file:
            pickle.dump((self, labels), file)

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
        nodes = {
            KEY: list(range(self.graph.num_nodes(KEY))),
            VALUE: self.graph.nodes[VALUE].data[SID] == 0,
        }
        for ntype in self.graph.ntypes:
            if ntype in nodes:
                continue
            nodes[ntype] = [i]
        graph = dgl.node_subgraph(self.graph, nodes=nodes)
        return Graph(graph)

    @classmethod
    def batch(cls, graphs):
        return cls(dgl.batch([graph.graph for graph in graphs]))

    @classmethod
    def reduce(cls, **kwargs):
        raise NotImplementedError
