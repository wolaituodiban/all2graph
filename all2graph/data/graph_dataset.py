import numpy as np

import torch
from torch.utils.data import Dataset

from ..utils.dgl_utils import dgl
from ..utils import progress_wrapper


class GraphDataset(Dataset):
    def __init__(self, paths, partitions=1, disable=True):
        self.paths = []
        for path in progress_wrapper(paths, disable=disable, postfix='checking files'):
            try:
                (meta_graphs, graphs), labels = dgl.load_graphs(path)
                assert isinstance(labels, dict)
                component_ids = np.unique(meta_graphs.ndata['component_id'])
                for k, v in labels.items():
                    assert v.shape[0] == component_ids.shape[0]
                if partitions == 1:
                    self.paths.append((path, 0, np.inf))
                else:
                    num_per_part = int(np.floor(component_ids.shape[0] / partitions))
                    for i in range(int(np.floor(component_ids.shape[0] / num_per_part))):
                        self.paths.append((path, i*num_per_part, (i+1)*num_per_part))
            except:
                print('{}加载失败'.format(path))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path, lower, upper = self.paths[item]
        (meta_graphs, graphs), labels = dgl.load_graphs(path)
        if upper != np.inf:
            mask = (lower <= meta_graphs.ndata['component_id']) & (meta_graphs.ndata['component_id'] < upper)
            meta_graphs = dgl.node_subgraph(meta_graphs, mask)

            min_id = meta_graphs.ndata[dgl.NID].min()
            max_id = meta_graphs.ndata[dgl.NID].max()
            mask = (min_id <= graphs.ndata['meta_node_id']) & (graphs.ndata['meta_node_id'] < max_id)
            graphs = dgl.node_subgraph(graphs, mask, store_ids=True)
            labels = {k: v[mask] for k, v in labels.items()}
        return meta_graphs, graphs, labels

    @staticmethod
    def collate_fn(batches):
        meta_graphs = []
        graphs = []
        labels = {}
        for mg, g, l in batches:
            meta_graphs.append(mg)
            graphs.append(g)
            for k, v in l.items():
                if k not in labels:
                    labels[k] = [v]
                else:
                    labels[k].append(v)
        meta_graphs = dgl.batch(meta_graphs)
        graphs = dgl.batch(graphs)
        labels = {k: torch.stack(v) for k, v in labels.items()}
        return meta_graphs, graphs, labels
