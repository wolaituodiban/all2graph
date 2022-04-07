import gzip
import pickle
from typing import List, Dict

import dgl
import torch

from ..meta_struct import MetaStruct
from ..globals import *


class Graph(MetaStruct):
    def __init__(self, graph: dgl.DGLGraph, key_tensor: torch.Tensor, key_mapper: Dict[str, int],
                 targets: List[str], nodes_per_sample: List[int], **kwargs):
        """

        Args:
            graph:
            key_mapper:
            key_tensor: key对应的分词组成的张量
            targets: 目标keys
            nodes_per_sample: 拆分sample，每个值代表每个sample的num of nodes
            **kwargs:
        """
        super().__init__(initialized=True, **kwargs)
        self.graph = graph
        self.key_tensor = key_tensor
        self.key_mapper = key_mapper
        self.targets = targets
        self.nodes_per_sample = nodes_per_sample

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
        return len(self.nodes_per_sample)

    @property
    def keys(self):
        return self.graph.ndata[KEY]

    @property
    def strings(self):
        return self.graph.ndata[STRING]

    @property
    def numbers(self):
        return self.graph.ndata[NUMBER]

    @property
    def seq_ids(self):
        output = []
        unique_keys = torch.unique(self.keys).unsqueeze(1)
        for i, keys in enumerate(torch.split(self.keys, self.nodes_per_sample)):
            mask = unique_keys == keys.unsqueeze(0)
            for row in mask:
                ids = torch.flatten(torch.nonzero(row))
                if ids.shape[0] == 0:
                    continue
                if i > 0:
                    ids += sum(self.nodes_per_sample[:i])
                output.append(ids)
        return output

    @property
    def readout_ids(self):
        return torch.flatten(torch.nonzero(self.keys == self.key_mapper[READOUT]))

    def add_self_loop(self):
        self.graph = dgl.add_self_loop(self.graph)
        return self

    def to_simple(self, writeback_mapping=False, **kwargs):
        if writeback_mapping:
            self.graph, wm = dgl.to_simple(self.graph, writeback_mapping=writeback_mapping, **kwargs)
            return self, wm
        else:
            self.graph = dgl.to_simple(self.graph, writeback_mapping=writeback_mapping, **kwargs)
            return self

    def to_bidirectied(self, *args, **kwargs):
        self.graph = dgl.to_bidirected(self.graph, *args, **kwargs)
        return self

    def to(self, *args, **kwargs):
        self.graph = self.graph.to(*args, **kwargs)
        self.key_tensor = self.key_tensor.to(*args, **kwargs)
        return self

    def cpu(self, *args, **kwargs):
        self.graph = self.graph.cpu()
        self.key_tensor = self.key_tensor.cpu(*args, **kwargs)
        return self

    def pin_memory(self):
        if dgl.__version__ >= '0.8':
            self.graph = self.graph.pin_memory_()
        else:
            for k, v in self.graph.ndata.items():
                self.graph.ndata[k] = v.pin_memory()
            for k, v in self.graph.edata.items():
                self.graph.edata[k] = v.pin_memory()
        self.key_tensor = self.key_tensor.pin_memory()
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
