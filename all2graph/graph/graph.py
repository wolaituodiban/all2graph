import gzip
import pickle

import dgl
import torch

from ..globals import *
from ..meta_struct import MetaStruct


class Graph(MetaStruct):
    def __init__(self, graph: dgl.DGLGraph, seq2node: torch.Tensor, seq_type: torch.Tensor, seq_sample: torch.Tensor,
                 type_string: torch.Tensor, targets: torch.Tensor, readout: int, **kwargs):
        """

        Args:
            graph: 
                ndata:
                    string: long (*, )
                    number: float32 (*, )
                    sequence: long (*, 2)
            seq2node: 序列到图的映射关系
            seq_type: 序列类型
            seq_sample: 序列对应的样本编号
            type_string: 类型对应的字符串编码
            targets: 目标对应的类型
            readout: 对应的类型
            **kwargs:
        """
        super().__init__(initialized=True, **kwargs)
        self.graph = graph
        self.seq2node = seq2node
        self.seq_type = seq_type
        self.seq_sample = seq_sample
        self.type_string = type_string
        self.targets = targets
        self.readout = readout

    def __repr__(self):
        return self.graph.__repr__()

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
    def node2seq(self):
        return self.graph.ndata[SEQUENCE]

    @property
    def types(self):
        return self.seq_sample[self.node2seq[:, 0]]

    @property
    def readout_mask(self):
        return self.types == self.readout

    def add_self_loop(self):
        graph = dgl.add_self_loop(self.graph)
        return Graph(graph, seq2node=self.seq2node, seq_type=self.seq_type, seq_sample=self.seq_sample,
                     type_string=self.type_string, targets=self.targets, readout=self.readout)

    def to_simple(self, writeback_mapping=False, **kwargs):
        if writeback_mapping:
            graph, wm = dgl.to_simple(self.graph, writeback_mapping=writeback_mapping, **kwargs)
            graph = Graph(graph, seq2node=self.seq2node, seq_type=self.seq_type, seq_sample=self.seq_sample,
                          type_string=self.type_string, targets=self.targets, readout=self.readout)
            return graph, wm
        else:
            graph = dgl.to_simple(self.graph, writeback_mapping=writeback_mapping, **kwargs)
            return Graph(graph, seq2node=self.seq2node, seq_type=self.seq_type, seq_sample=self.seq_sample,
                         type_string=self.type_string, targets=self.targets, readout=self.readout)

    def to_bidirectied(self, *args, **kwargs):
        graph = dgl.to_bidirected(self.graph, *args, **kwargs)
        return Graph(graph, seq2node=self.seq2node, seq_type=self.seq_type, seq_sample=self.seq_sample,
                     type_string=self.type_string, targets=self.targets, readout=self.readout)

    def to(self, *args, **kwargs):
        self.graph.to(*args, **kwargs)
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                v.to(*args, **kwargs)
        return self

    def cpu(self, *args, **kwargs):
        self.graph.cpu()
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                v.cpu(*args, **kwargs)
        return self

    def pin_memory(self):
        if dgl.__version__ >= '0.8':
            self.graph.pin_memory_()
        else:
            for k, v in self.graph.ndata.items():
                v.pin_memory()
            for k, v in self.graph.edata.items():
                v.pin_memory()
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                v.pin_memory()
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
