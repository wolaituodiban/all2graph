import sys
import traceback
from typing import List, Union, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ...factory import Factory
from ...utils import progress_wrapper


class GraphDataset(Dataset):
    def __init__(self, paths, factory: Factory, partitions=1, shuffle=False, disable=True, **kwargs):
        partitions = int(partitions)
        self.factory = factory
        self.kwargs = kwargs
        self.paths: List[Tuple[str, Union[torch.Tensor, None]]] = []  # (路径，分片id)
        for path in progress_wrapper(paths, disable=disable, postfix='checking files'):
            try:
                df = pd.read_csv(path, **self.kwargs)
                assert set(self.factory.graph_parser.targets) < set(df.columns)
                unique_component_ids = np.arange(0, df.shape[0], 1)
            except:
                print(path, file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                continue
            if partitions == 1:
                self.paths.append((path, None))
            else:
                for ids in self.split_index(unique_component_ids, partitions, shuffle):
                    self.paths.append((path, ids))

    @staticmethod
    def split_index(indices, partitions, shuffle) -> List[torch.Tensor]:
        num = indices.shape[0]
        remain_number = num % partitions
        if remain_number != 0:
            padding = [np.nan] * (partitions - remain_number)
            indices = np.concatenate([indices, padding])
            num = indices.shape[0]
        if shuffle:
            rank = np.argsort(np.random.random(num))
            indices = indices[rank]
        splits = []
        for ids in np.split(indices, partitions):
            ids = ids[np.bitwise_not(np.isnan(ids))]
            if ids.shape[0] > 0:
                ids = np.sort(ids)
                splits.append(torch.tensor(ids))
        return splits

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item) -> pd.DataFrame:
        path, component_ids = self.paths[item]
        df = pd.read_csv(path, **self.kwargs)
        if component_ids is not None:
            df = df.iloc[component_ids]
        return df

    def collate_fn(self, batches):
        df = pd.concat(batches)
        return self.factory.produce_dgl_graph_and_label(df)
