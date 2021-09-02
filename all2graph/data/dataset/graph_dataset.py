import numpy as np

import torch
from torch.utils.data import Dataset

from all2graph.macro import COMPONENT_IDS, META_NODE_IDS, META_EDGE_IDS
from all2graph.utils.dgl_utils import dgl
from all2graph.utils import progress_wrapper


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
            unique_component_ids = np.unique(meta_graphs.ndata[COMPONENT_IDS])
            num_components = unique_component_ids.shape[0]
            for k, v in labels.items():
                assert v.shape[0] == num_components
            if partitions == 1:
                self.paths.append((path, None))
            else:
                remain_number = num_components % partitions
                if remain_number != 0:
                    padding = [np.nan] * (partitions - remain_number)
                    unique_component_ids = np.concatenate([unique_component_ids, padding])
                    num_components = unique_component_ids.shape[0]
                if shuffle:
                    rank = np.argsort(np.random.random(num_components))
                    unique_component_ids = unique_component_ids[rank]
                for ids in np.split(unique_component_ids, partitions):
                    ids = ids[np.bitwise_not(np.isnan(ids))]
                    ids = np.sort(ids)
                    ids = torch.tensor(ids, dtype=torch.long)
                    if ids.shape[0] > 0:
                        self.paths.append((path, ids))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path, component_ids = self.paths[item]
        (meta_graphs, graphs), labels = dgl.load_graphs(path)
        if component_ids is not None:
            meta_graphs_mask = (meta_graphs.ndata[COMPONENT_IDS].view(-1, 1) == component_ids).any(1)
            meta_graphs = dgl.node_subgraph(meta_graphs, meta_graphs_mask)

            graphs_mask = (graphs.ndata[COMPONENT_IDS].view(-1, 1) == component_ids).any(1)
            graphs = dgl.node_subgraph(graphs, graphs_mask)

            min_component_ids = component_ids.min()
            meta_graphs.ndata[COMPONENT_IDS] -= min_component_ids
            graphs.ndata[COMPONENT_IDS] -= min_component_ids

            min_meta_node_ids = meta_graphs.ndata[META_NODE_IDS].min()
            meta_graphs.ndata[META_NODE_IDS] -= min_meta_node_ids
            graphs.ndata[META_NODE_IDS] -= min_meta_node_ids

            if meta_graphs.num_edges() > 0:
                min_meta_edge_ids = meta_graphs.edata[META_EDGE_IDS].min()
                meta_graphs.edata[META_EDGE_IDS] -= min_meta_edge_ids
                graphs.edata[META_EDGE_IDS] -= min_meta_edge_ids

            labels = {k: v[component_ids] for k, v in labels.items()}

        return meta_graphs, graphs, labels

    @staticmethod
    def collate_fn(batches):
        meta_graphss = []
        graphss = []
        labelss = {}
        max_component_id = 0
        max_meta_node_id = 0
        max_edge_node_id = 0
        for meta_graphs, graphs, labels in batches:
            meta_graphs.ndata[COMPONENT_IDS] += max_component_id
            graphs.ndata[COMPONENT_IDS] += max_component_id
            max_component_id = meta_graphs.ndata[COMPONENT_IDS].max() + 1

            meta_graphs.ndata[META_NODE_IDS] += max_meta_node_id
            graphs.ndata[META_NODE_IDS] += max_meta_node_id
            max_meta_node_id = meta_graphs.ndata[META_NODE_IDS].max() + 1

            if meta_graphs.num_edges() > 0:
                meta_graphs.edata[META_EDGE_IDS] += max_edge_node_id
                graphs.edata[META_EDGE_IDS] += max_edge_node_id
                max_edge_node_id += meta_graphs.edata[META_EDGE_IDS].max() + 1

            meta_graphss.append(meta_graphs)
            graphss.append(graphs)
            for k, v in labels.items():
                if k not in labelss:
                    labelss[k] = [v]
                else:
                    labelss[k].append(v)
        meta_graphs = dgl.batch(meta_graphss)
        graphs = dgl.batch(graphss)
        labels = {k: torch.stack(v) for k, v in labelss.items()}
        return meta_graphs, graphs, labels
