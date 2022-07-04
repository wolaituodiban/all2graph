from audioop import add
import gzip
import pickle
from typing import Dict, List

import dgl
import torch

from ..globals import *
from ..meta_struct import MetaStruct


class Graph(MetaStruct):
    def __init__(self, graph: dgl.DGLGraph, seq_type: torch.Tensor, seq_sample: torch.Tensor,
                 type_string: torch.Tensor, targets: List[str], type_mapper: Dict[str, int], **kwargs):
        """

        Args:
            graph: 
                ndata:
                    string: long (*, )
                    number: float32 (*, )
                    sequence: long (*, 2)
            node2seq: 节点到序列的映射关系
            seq_type: 序列类型
            seq_sample: 序列对应的样本编号
            type_string: 类型对应的字符串编码
            targets: 目标对应的类型
            type_mapper:
            **kwargs:
        """
        super().__init__(**kwargs)
        self.graph = graph
        self.seq_type = seq_type
        self.seq_sample = seq_sample
        self.type_string = type_string
        self.targets = targets
        self.type_mapper = type_mapper

    def __repr__(self):
        return self.graph.__repr__()

    @property
    def readout(self):
        return self.type_mapper[READOUT]

    @property
    def device(self):
        return self.graph.device

    @property
    def nodes(self):
        return self.graph.nodes

    @property
    def edges(self):
        return self.graph.edges

    @property
    def ndata(self):
        return self.graph.ndata

    @property
    def num_nodes(self):
        return self.graph.num_nodes()

    @property
    def num_edges(self):
        return self.graph.num_edges()

    @property
    def num_samples(self):
        return torch.unique(self.seq_sample).shape[0]

    @property
    def strings(self):
        return self.graph.ndata[STRING]

    @property
    def numbers(self):
        return self.graph.ndata[NUMBER]

    @property
    def types(self):
        return self.seq_type[self.seq2node()[0]]

    @property
    def readout_mask(self):
        return self.types == self.readout

    @property
    def num_seqs(self):
        return self.seq_type.shape[0]

    @property
    def node2seq(self) -> torch.Tensor:
        # 序列和图需要保证双向映射
        # 然而序列需要padding
        # 实验显示，如果padding同一个值，会导致backward在同一个位置多次叠加梯度，进而降低速度
        # 因此做了随机padding的优化，并设置为负数，方便后续mask
        ind1, ind2 = self.seq2node()
        node2seq = torch.randint(
            low=-self.num_nodes, high=0, size=(ind1.max() + 1, ind2.max() + 1), device=self.device)
        node2seq[ind1, ind2] = torch.arange(self.num_nodes, device=self.device)
        return node2seq

    def seq2node(self, dim=None):
        ind1, ind2 = self.graph.ndata[SEQ2NODE][:, 0], self.graph.ndata[SEQ2NODE][:, 1]
        if dim is None:
            return ind1, ind2
        else:
            ind1 = ind1.unsqueeze(-1).expand(-1, dim)
            ind2 = ind2.unsqueeze(-1).expand(-1, dim)
            ind3 = torch.arange(dim, device=self.device).expand(self.num_nodes, -1)
            return ind1, ind2, ind3

    def seq_mask(self, types: List[str]):
        if types:
            types = [self.type_mapper[t] for t in types if t in self.type_mapper]
            types = torch.tensor(types, dtype=torch.long, device=self.device)
            return (self.seq_type.unsqueeze(1) == types.unsqueeze(0)).any(1)
        else:
            return None

    def add_self_loop(self):
        graph = dgl.add_self_loop(self.graph)
        return Graph(graph,  seq_type=self.seq_type, seq_sample=self.seq_sample,
                     type_string=self.type_string, targets=self.targets,
                     type_mapper=self.type_mapper)

    def to_simple(self, writeback_mapping=False, **kwargs):
        if writeback_mapping:
            graph, wm = dgl.to_simple(self.graph, writeback_mapping=writeback_mapping, **kwargs)
            graph = Graph(graph,  seq_type=self.seq_type, seq_sample=self.seq_sample,
                          type_string=self.type_string, targets=self.targets, type_mapper=self.type_mapper)
            return graph, wm
        else:
            graph = dgl.to_simple(self.graph, writeback_mapping=writeback_mapping, **kwargs)
            return Graph(graph, seq_type=self.seq_type, seq_sample=self.seq_sample,
                         type_string=self.type_string, targets=self.targets, type_mapper=self.type_mapper)

    def to_bidirected(self, *args, **kwargs):
        graph = dgl.to_bidirected(self.graph, *args, **kwargs)
        return Graph(graph, seq_type=self.seq_type, seq_sample=self.seq_sample,
                     type_string=self.type_string, targets=self.targets, type_mapper=self.type_mapper)

    def add_edges_by_seq(self, degree, r_degree, add_self_loop=False):
        all_u, all_v = [], []
        if add_self_loop:
            nodes = torch.arange(self.graph.num_nodes(), device=self.device)
            all_u.append(nodes)
            all_v.append(nodes)
        seq, node = torch.sort(self.seq2node()[0], stable=True)
        for d in range(1, max(degree, r_degree) + 1):
            u = torch.nonzero(seq[:-d] == seq[d:]).squeeze(-1)
            v = u + d
            u = node[u]
            v = node[v]
            if d <= degree:
                all_u.append(u)
                all_v.append(v)
            if d <= r_degree:
                all_u.append(v)
                all_v.append(u)
        all_u = torch.cat(all_u)
        all_v = torch.cat(all_v)
        #     assert (self.seq2node()[0][all_u] == self.seq2node()[0][all_v]).all()
        graph = dgl.add_edges(self.graph, all_u, all_v)
        return Graph(graph, seq_type=self.seq_type, seq_sample=self.seq_sample,
                     type_string=self.type_string, targets=self.targets, type_mapper=self.type_mapper)

    def to(self, *args, **kwargs):
        self.graph = self.graph.to(*args, **kwargs)
        self.seq_type = self.seq_type.to(*args, **kwargs)
        self.seq_sample = self.seq_sample.to(*args, **kwargs)
        self.type_string = self.type_string.to(*args, **kwargs)
        return self

    def cpu(self, *args, **kwargs):
        self.graph = self.graph.cpu()
        self.seq_type = self.seq_type.cpu(*args, **kwargs)
        self.seq_sample = self.seq_sample.cpu(*args, **kwargs)
        self.type_string = self.type_string.cpu(*args, **kwargs)
        return self

    def pin_memory(self):
        if dgl.__version__ >= '0.8':
            self.graph.pin_memory_()
        else:
            for k, v in self.graph.ndata.items():
                self.graph.ndata[k] = v.pin_memory()
            for k, v in self.graph.edata.items():
                self.graph.edata[k] = v.pin_memory()
        self.seq_type = self.seq_type.pin_memory()
        self.seq_sample = self.seq_sample.pin_memory()
        self.type_string = self.type_string.pin_memory()
        return self

    def local_scope(self):
        return self.graph.local_scope()

    @classmethod
    def load(cls, path, **kwargs):
        with gzip.open(path, 'rb', **kwargs) as file:
            return pickle.load(file)

    def save(self, path, labels, **kwargs):
        with gzip.open(path, 'wb', **kwargs) as file:
            pickle.dump((self, labels), file)
