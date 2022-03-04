import gzip
import pickle

import dgl
import dgl.function as fn
import torch
import pandas as pd

from .raw_graph import gen_edges_for_seq_
from ..globals import KEY, VALUE, TOKEN, NUMBER, KEY2VALUE, EDGE, KEY2KEY, VALUE2VALUE, SAMPLE, SAMPLE2VALUE
from ..meta_struct import MetaStruct


def tensor2list(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy().tolist()


class Graph(MetaStruct):
    def __init__(self, graph: dgl.DGLHeteroGraph, **kwargs):
        super().__init__(initialized=True, **kwargs)
        self.graph = graph

    @classmethod
    def from_data(cls, edges, num_samples, key_tokens, value_tokens, numbers, **kwargs):
        num_nodes_dict = {SAMPLE: num_samples, KEY: key_tokens.shape[0], VALUE: value_tokens.shape[0]}
        for (_, _, vtype), (u, v) in edges.items():
            if vtype in num_nodes_dict:
                continue
            num_nodes_dict[vtype] = len(v)
        graph = dgl.heterograph(edges, num_nodes_dict=num_nodes_dict)
        graph.nodes[KEY].data[TOKEN] = key_tokens
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
        return {ntype for ntype in self.graph.ntypes if ntype != KEY and ntype != VALUE and ntype != SAMPLE}

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

    def to_bidirectied(self, etype=None):
        raise NotImplementedError

    def add_edges_by_key(self, degree, r_degree, keys=None):
        all_u, all_v = [], []

        sid = self.graph.adj(transpose=True, etype=SAMPLE2VALUE).coalesce().indices()[1]
        kid = self.graph.adj(transpose=True, etype=KEY2VALUE).coalesce().indices()[1]
        df = pd.DataFrame({'sid': sid.cpu().detach().numpy(), 'kid': kid.cpu().detach().numpy()})
        for (sid, kid), group in df.groupby(['sid', 'kid']):
            if keys and kid not in keys:
                continue
            u, v = gen_edges_for_seq_(group.index.tolist(), degree=degree, r_degree=r_degree)
            all_u += u
            all_v += v
        all_u = torch.tensor(all_u, dtype=torch.long)
        all_v = torch.tensor(all_v, dtype=torch.long)

        graph = dgl.add_edges(
            self.graph, torch.tensor(all_u, dtype=torch.long), torch.tensor(all_v, dtype=torch.long), etype=VALUE2VALUE)
        return Graph(graph)

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

    def sample_subgraph(self, i):
        nodes = {
            KEY: list(range(self.graph.num_nodes(KEY))),
            SAMPLE: i
        }
        for etype in self.graph.canonical_etypes:
            if etype[0] == SAMPLE:
                nodes[etype[-1]] = self.graph.successors(i, etype=etype)
        graph = dgl.node_subgraph(self.graph, nodes=nodes)
        return Graph(graph)

    @classmethod
    def batch(cls, graphs):
        return cls(dgl.batch([graph.graph for graph in graphs]))
