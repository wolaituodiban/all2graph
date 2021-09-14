import sys
import traceback
from typing import List, Union, Tuple

import numpy as np
import pandas as pd
import dgl
import torch
from torch.utils.data import Dataset

from ...globals import COMPONENT_ID
from ...factory import Factory
from ...utils import progress_wrapper


class GraphDataset(Dataset):
    def __init__(self, paths, factory: Factory, partitions=1, shuffle=False, disable=True, **kwargs):
        partitions = int(partitions)
        self.factory = factory
        self.kwargs = kwargs
        self.paths: List[Tuple[str, Union[torch.Tensor, None], bool]] = []  # (路径，分片id，是否是图文件)
        for path in progress_wrapper(paths, disable=disable, postfix='checking files'):
            try:
                (meta_graphs, graphs), labels = dgl.load_graphs(path)
                assert isinstance(labels, dict)
                unique_component_ids = np.unique(meta_graphs.ndata[COMPONENT_ID].abs())
                num_components = unique_component_ids.shape[0]
                for k, v in labels.items():
                    assert v.shape[0] == num_components
                is_graph_file = True
            except dgl.DGLError:
                df = pd.read_csv(path, **self.kwargs)
                assert set(self.factory.data_parser.target_cols) < set(df.columns)
                is_graph_file = False
                unique_component_ids = np.arange(0, df.shape[0], 1)
            except:
                print(path, file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                continue
            if partitions == 1:
                self.paths.append((path, None, is_graph_file))
            else:
                for ids in self.split_index(unique_component_ids, partitions, shuffle):
                    self.paths.append((path, ids, is_graph_file))

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

    def __getitem__(self, item):
        path, component_ids, is_graph_file = self.paths[item]
        if is_graph_file:
            return self.factory.load_graphs(path, component_ids)
        else:
            df = pd.read_csv(path, **self.kwargs)
            if component_ids is not None:
                df = df.iloc[component_ids]
            return self.factory.product_graphs(df)
