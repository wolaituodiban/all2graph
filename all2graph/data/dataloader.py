import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from ..graph import RawGraph
from ..parsers import RawGraphParser


class DataLoader(TorchDataLoader):
    def __init__(self, dataset: Dataset, parser: RawGraphParser = None, **kwargs):
        self.parser = parser
        super().__init__(dataset, collate_fn=self.collate_fn, **kwargs)

    def collate_fn(self, batches):
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
