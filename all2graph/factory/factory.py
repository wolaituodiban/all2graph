import os
from multiprocessing import Pool
from typing import Iterable, Tuple, List, Union

import dgl
import pandas as pd
import torch

from ..globals import COMPONENT_ID, META_NODE_ID, META_EDGE_ID
from ..meta import MetaInfo
from ..data import DataParser
from ..graph.graph_transer import GraphTranser
from ..utils import progress_wrapper
from ..utils.pd_utils import dataframe_chunk_iter


class Factory:
    def __init__(self, data_parser: DataParser,
                 meta_graph_config: dict = None, graph_transer_config: dict = None):
        self.data_parser = data_parser
        self.meta_graph_config = meta_graph_config or {}
        self.graph_transer_config = graph_transer_config or {}
        self.graph_transer: Union[GraphTranser, None] = None
        self.save_path = None

    def _produce_meta_graph(self, chunk: pd.DataFrame) -> Tuple[MetaInfo, int]:
        graph, global_index_mapper, local_index_mappers = self.data_parser.parse(chunk, progress_bar=False)
        index_ids = list(global_index_mapper.values())
        for mapper in local_index_mappers:
            index_ids += list(mapper.values())
        meta_graph = MetaInfo.from_data(graph, index_nodes=index_ids, progress_bar=False, **self.meta_graph_config)
        return meta_graph, chunk.shape[0]

    def produce_meta_graph(self, data: Union[pd.DataFrame, Iterable[pd.DataFrame]], chunksize=64, progress_bar=False,
                           postfix='reading csv', processes=0, **kwargs) -> MetaInfo:
        if isinstance(data, (str, pd.DataFrame)):
            data = dataframe_chunk_iter(data, chunksize=chunksize, **kwargs)

        meta_graphs: List[MetaInfo] = []
        weights = []
        if processes == 0:
            results = map(self._produce_meta_graph, data)
            results = progress_wrapper(results, disable=not progress_bar, postfix=postfix)
            for meta_graph, weight in results:
                meta_graphs.append(meta_graph)
                weights.append(weight)
        else:
            with Pool(processes) as pool:
                results = pool.imap(self._produce_meta_graph, data)
                results = progress_wrapper(results, disable=not progress_bar, postfix=postfix)
                for meta_graph, weight in results:
                    meta_graphs.append(meta_graph)
                    weights.append(weight)

        meta_graph = MetaInfo.reduce(
            meta_graphs, weights=weights, progress_bar=progress_bar, processes=processes, **self.meta_graph_config
        )
        self.graph_transer = GraphTranser.from_data(meta_graph, **self.graph_transer_config)
        return meta_graph

    def product_graphs(self, chunk: pd.DataFrame):
        graph, *_ = self.data_parser.parse(chunk, progress_bar=False)
        dgl_meta_graph, dgl_graph = self.graph_transer.graph_to_dgl(graph)
        labels = self.data_parser.gen_targets(chunk)
        return (dgl_meta_graph, dgl_graph), labels

    def _save_graph(self, chunk: pd.DataFrame):
        file_path = os.path.join(self.save_path, '{}.dgl.graph'.format(chunk.index[0]))
        (dgl_meta_graph, dgl_graph), labels = self.product_graphs(chunk)
        dgl.save_graphs(file_path, [dgl_meta_graph, dgl_graph], labels=labels)
        return file_path

    def save_graphs(self, data: Union[pd.DataFrame, Iterable[pd.DataFrame]], save_path,
                    chunksize=64, progress_bar=False, postfix='saving graphs', processes=0, **kwargs):
        self.save_path = save_path
        if isinstance(data, (str, pd.DataFrame)):
            data = dataframe_chunk_iter(data, chunksize=chunksize, **kwargs)
        if processes == 0:
            list(progress_wrapper(map(self._save_graph, data), disable=not progress_bar, postfix=postfix))
        else:
            with Pool(processes) as pool:
                temp = pool.imap(self._save_graph, data)
                list(progress_wrapper(temp, disable=not progress_bar, postfix=postfix))

    @staticmethod
    def load_graphs(path, component_ids: torch.Tensor):
        (meta_graph, graph), labels = dgl.load_graphs(path)
        if component_ids is not None:
            graph_mask = (meta_graph.ndata[COMPONENT_ID][graph.ndata[META_NODE_ID]].view(-1, 1) == component_ids).any(1)
            graph = dgl.node_subgraph(graph, graph_mask)

            meta_graph_mask = (meta_graph.ndata[COMPONENT_ID].view(-1, 1) == component_ids).any(1)
            meta_graph = dgl.node_subgraph(meta_graph, meta_graph_mask)

            labels = {k: v[component_ids] for k, v in labels.items()}
        return (meta_graph, graph), labels

    @staticmethod
    def batch(batches):
        if len(batches) == 1:
            return batches[0]
        # 为了防止id冲突，要给每个小图加上之前id的最大值
        meta_graphss = []
        graphss = []
        labelss = {}
        max_component_id = 0
        max_meta_node_id = 0
        max_edge_node_id = 0
        for (meta_graphs, graphs), labels in batches:
            # 当COMPONENT_ID为负数的时候，应该加上max_component_id的相反数
            meta_graphs.ndata[COMPONENT_ID] += max_component_id
            max_component_id = meta_graphs.ndata[COMPONENT_ID].max() + 1

            meta_graphs.ndata[META_NODE_ID] += max_meta_node_id
            graphs.ndata[META_NODE_ID] += max_meta_node_id
            max_meta_node_id = meta_graphs.ndata[META_NODE_ID].max() + 1

            if meta_graphs.num_edges() > 0:
                meta_graphs.edata[META_EDGE_ID] += max_edge_node_id
                graphs.edata[META_EDGE_ID] += max_edge_node_id
                max_edge_node_id += meta_graphs.edata[META_EDGE_ID].max() + 1

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
        return (meta_graphs, graphs), labels
