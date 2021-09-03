from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch

from .graph_dataset import GraphDataset
from ..data_parser import DataParser
from ...graph.graph_transer import GraphTranser
from ...utils import progress_wrapper


class RealtimeDataset(GraphDataset):
    def __init__(self, paths, preprocessor: Callable[[pd.DataFrame], Iterable], data_parser: DataParser,
                 graph_transer: GraphTranser, label_cols=None, partitions=1, shuffle=False, disable=True, **kwargs):
        super().__init__([])
        self.paths = paths
        self.preprocessor = preprocessor
        self.data_parser = data_parser
        self.graph_transer = graph_transer
        self.label_cols = list(label_cols or [])

        self.kwargs = kwargs

        partitions = int(partitions)
        self.paths = []
        for path in progress_wrapper(paths, disable=disable, postfix='checking files'):
            df = pd.read_csv(path, **self.kwargs)

            if partitions == 1:
                self.paths.append((path, None))
            else:
                indices = np.arange(0, df.shape[0], 1)
                for ids in self.split_index(indices, partitions, shuffle):
                    self.paths.append((path, ids))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path, row_num = self.paths[item]
        df = pd.read_csv(path, **self.kwargs)
        if row_num is not None:
            df = df.iloc[row_num]
        iterable = self.preprocessor(df)
        graph, *_ = self.data_parser.parse(iterable, False)
        meta_graph, graph = self.graph_transer.graph_to_dgl(graph)
        labels = {
            col: torch.tensor(pd.to_numeric(df[col].values, errors='coerce'), dtype=torch.float32)
            for col in self.label_cols if col in df
        }
        return meta_graph, graph, labels
