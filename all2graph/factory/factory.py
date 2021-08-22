import sys
from multiprocessing import Pool
from typing import Callable, Iterable, Tuple, List

import pandas as pd
import toad
from toad.utils.progress import Progress

from ..utils import dataframe_chunk_iter
from ..meta_graph import MetaGraph
from ..resolver import Resolver


class Factory:
    def __init__(self, resolver: Resolver, preprocessor: Callable[[pd.DataFrame], Iterable], **kwargs):
        self.preprocessor = preprocessor
        self.resolver = resolver
        self.transformer_config = kwargs
        self.meta_graph_config = None

    def _produce_meta_graph(self, chunk: pd.DataFrame) -> Tuple[MetaGraph, int]:
        data = self.preprocessor(chunk)
        graph, global_index_mapper, local_index_mappers = self.resolver.resolve(data, progress_bar=False)
        index_ids = list(global_index_mapper.values())
        for mapper in local_index_mappers:
            index_ids += list(mapper.values())
        meta_graph = MetaGraph.from_data(graph, index_nodes=index_ids, progress_bar=False, **self.meta_graph_config)
        return meta_graph, chunk.shape[0]

    def produce(
            self, data: Iterable[pd.DataFrame], progress_bar=False, suffix='reading csv',
            processes=None, chunksize=1, read_csv_kwargs=None, **meta_graph_kwargs
    ) -> MetaGraph:
        self.meta_graph_config = meta_graph_kwargs
        read_csv_kwargs = read_csv_kwargs or {}
        if isinstance(data, (str, pd.DataFrame)):
            data = dataframe_chunk_iter(data, **read_csv_kwargs)
        if progress_bar and not isinstance(data, Progress):
            data = Progress(data)
            if toad.version.__version__ <= '0.0.65' and data.size is None:
                data.size = sys.maxsize
            data.suffix = suffix

        meta_graphs: List[MetaGraph] = []
        weights = []
        if processes == 0:
            for meta_graph, weight in map(self._produce_meta_graph, data):
                meta_graphs.append(meta_graph)
                weights.append(weight)
        else:
            with Pool(processes) as pool:
                for meta_graph, weight in pool.imap(self._produce_meta_graph, data, chunksize=chunksize):
                    meta_graphs.append(meta_graph)
                    weights.append(weight)

        meta_graph = MetaGraph.reduce(meta_graphs, weights=weights, progress_bar=progress_bar, **meta_graph_kwargs)
        return meta_graph
