import os
from multiprocessing import Pool
from typing import Callable, Iterable, Tuple, List, Union

import pandas as pd
import torch

from ..meta import MetaGraph
from ..data import DataParser
from ..graph.graph_transer import GraphTranser
from ..utils import progress_wrapper
from ..utils.pd_utils import dataframe_chunk_iter
from ..utils.dgl_utils import dgl


class Factory:
    def __init__(self, preprocessor: Callable[[pd.DataFrame], Iterable], data_parser: DataParser,
                 meta_graph_config: dict = None, graph_transer_config: dict = None):
        self.preprocessor = preprocessor
        self.data_parser = data_parser
        self.meta_graph_config = meta_graph_config or {}
        self.graph_transer_config = graph_transer_config or {}
        self.graph_transer: Union[GraphTranser, None] = None
        self.label_cols = None
        self.save_path = None

    def _produce_meta_graph(self, chunk: pd.DataFrame) -> Tuple[MetaGraph, int]:
        data = self.preprocessor(chunk)
        graph, global_index_mapper, local_index_mappers = self.data_parser.parse(data, progress_bar=False)
        index_ids = list(global_index_mapper.values())
        for mapper in local_index_mappers:
            index_ids += list(mapper.values())
        meta_graph = MetaGraph.from_data(graph, index_nodes=index_ids, progress_bar=False, **self.meta_graph_config)
        return meta_graph, chunk.shape[0]

    def produce_meta_graph(self, data: Union[pd.DataFrame, Iterable[pd.DataFrame]], chunksize=64, progress_bar=False,
                           postfix='reading csv', processes=0, **kwargs) -> MetaGraph:
        if isinstance(data, (str, pd.DataFrame)):
            data = dataframe_chunk_iter(data, chunksize=chunksize, **kwargs)

        meta_graphs: List[MetaGraph] = []
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

        meta_graph = MetaGraph.reduce(
            meta_graphs, weights=weights, progress_bar=progress_bar, processes=processes, **self.meta_graph_config
        )
        self.graph_transer = GraphTranser.from_data(meta_graph, **self.graph_transer_config)
        return meta_graph

    def _save_graph(self, chunk: pd.DataFrame):
        data = self.preprocessor(chunk)
        graph, *_ = self.data_parser.parse(data, progress_bar=False)
        dgl_meta_graph, dgl_graph = self.graph_transer.graph_to_dgl(graph)
        file_path = os.path.join(self.save_path, '{}.dgl.graph'.format(chunk.index[0]))
        labels = {}
        for col in self.label_cols:
            if col in chunk:
                labels[col] = torch.tensor(pd.to_numeric(chunk[col], errors='coerce'), dtype=torch.float32)
        dgl.save_graphs(file_path, [dgl_meta_graph, dgl_graph], labels=labels)
        return file_path

    def save_graphs(self, data: Union[pd.DataFrame, Iterable[pd.DataFrame]], save_path, label_cols=None,
                    chunksize=64, progress_bar=False, postfix='saving graphs', processes=0, **kwargs):
        self.label_cols = label_cols or []
        self.save_path = save_path
        if isinstance(data, (str, pd.DataFrame)):
            data = dataframe_chunk_iter(data, chunksize=chunksize, **kwargs)
        if processes == 0:
            list(progress_wrapper(map(self._save_graph, data), disable=not progress_bar, postfix=postfix))
        else:
            with Pool(processes) as pool:
                temp = pool.imap(self._save_graph, data)
                list(progress_wrapper(temp, disable=not progress_bar, postfix=postfix))
