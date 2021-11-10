import os
import time
from multiprocessing import Pool
from typing import Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
from ..graph import RawGraph, Graph
from ..parsers import RawGraphParser


class DataLoader(TorchDataLoader):
    def __init__(self, dataset: TorchDataset, parser: RawGraphParser = None, temp_file=None, **kwargs):
        super().__init__(dataset, collate_fn=self.collate_fn, **kwargs)
        self.parser = parser

    def collate_fn(self, batches) -> Tuple[Graph, Dict[str, torch.Tensor]]:
        graphs = []
        labels = {}
        for graph, label in batches:
            graphs.append(graph)
            for k, v in label.items():
                if k in labels:
                    labels[k].append(v)
                else:
                    labels[k] = [v]
        graph = RawGraph.batch(graphs)
        labels = {k: torch.cat(v) for k, v in labels.items()}
        if self.parser is not None:
            graph = self.parser.parse(graph)
        return graph, labels

    def set_filter_key(self, x):
        self.parser.set_filter_key(x)


class DataLoaderV2(DataLoader):
    def __init__(
            self, dataset: TorchDataset, parser: RawGraphParser, num_workers=0, batch_size=1, shuffle=False,
            temp_file=False, **kwargs):
        super().__init__(dataset, parser, batch_size=batch_size)
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.temp_file = temp_file

    def parse(self, indices) -> Tuple[Graph, Dict[str, torch.Tensor]]:
        graph, labels = self.collate_fn([self.dataset[i] for i in indices])
        if self.temp_file:
            filename = str(time.time())
            filename = '+'.join([filename, str(id(filename))])+'.all2graph.graph'
            graph.save(filename)
            graph = filename
        return graph, labels

    def __iter__(self):
        """覆盖迭代器，防止torch.DataLoader的多进程会卡死的问题"""
        def foo(_graph):
            if isinstance(_graph, str):
                filename = _graph
                _graph = Graph.load(_graph)
                os.remove(filename)
            return _graph, labels

        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)
        indices = [indices[i*self.batch_size:(i+1)*self.batch_size] for i in range(len(self))]
        if self.num_workers == 0:
            for graph, labels in map(self.parse, indices):
                yield foo(graph)
        else:
            with Pool(self.num_workers) as pool:
                for graph, labels in pool.imap(self.parse, indices):
                    yield foo(graph)
