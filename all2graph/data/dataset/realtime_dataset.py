import numpy as np
import pandas as pd

from .graph_dataset import GraphDataset
from ...factory import Factory
from ...utils import progress_wrapper


class RealtimeDataset(GraphDataset):
    def __init__(self, paths, factory: Factory, partitions=1, shuffle=False, disable=True, **kwargs):
        super().__init__([])
        self.paths = paths
        self.factory = factory

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
        return self.factory.product_graphs(df)
