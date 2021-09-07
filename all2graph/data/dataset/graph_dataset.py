import numpy as np

import torch
from torch.utils.data import Dataset

from ...macro import COMPONENT_ID
from ...factory import Factory
from ...utils.dgl_utils import dgl
from ...utils import progress_wrapper


class GraphDataset(Dataset):
    def __init__(self, paths, partitions=1, shuffle=False, disable=True):
        partitions = int(partitions)
        self.paths = []
        for path in progress_wrapper(paths, disable=disable, postfix='checking files'):
            try:
                (meta_graphs, graphs), labels = dgl.load_graphs(path)
            except dgl.DGLError:
                print('{}加载失败'.format(path))
                continue
            assert isinstance(labels, dict)
            unique_component_ids = np.unique(meta_graphs.ndata[COMPONENT_ID])
            num_components = unique_component_ids.shape[0]
            for k, v in labels.items():
                assert v.shape[0] == num_components
            if partitions == 1:
                self.paths.append((path, None))
            else:
                for ids in self.split_index(unique_component_ids, partitions, shuffle):
                    self.paths.append((path, torch.tensor(ids, dtype=torch.long)))

    @staticmethod
    def split_index(indices, partitions, shuffle):
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
                splits.append(ids)
        return splits

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path, component_ids = self.paths[item]
        return Factory.load_graphs(path, component_ids)
