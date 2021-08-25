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
            self, data: Iterable[pd.DataFrame], chunksize=64, read_csv_kwargs=None, meta_graph_kwargs=None,
            progress_bar=False, suffix='reading csv', processes=0,
    ) -> MetaGraph:

        read_csv_kwargs = read_csv_kwargs or {}
        if isinstance(data, (str, pd.DataFrame)):
            data = dataframe_chunk_iter(data, chunksize=chunksize, **read_csv_kwargs)

        self.meta_graph_config = meta_graph_kwargs or {}
        meta_graphs: List[MetaGraph] = []
        weights = []
        if processes == 0:
            results = map(self._produce_meta_graph, data)
            if progress_bar and not isinstance(data, Progress):
                results = Progress(results)
                if toad.version.__version__ <= '0.0.65' and results.size is None:
                    results.size = sys.maxsize
                results.suffix = suffix
            for meta_graph, weight in results:
                meta_graphs.append(meta_graph)
                weights.append(weight)
        else:
            with Pool(processes) as pool:
                results = pool.imap(self._produce_meta_graph, data)
                if progress_bar and not isinstance(data, Progress):
                    results = Progress(results)
                    if toad.version.__version__ <= '0.0.65' and results.size is None:
                        results.size = sys.maxsize
                    results.suffix = suffix
                for meta_graph, weight in results:
                    meta_graphs.append(meta_graph)
                    weights.append(weight)

        meta_graph = MetaGraph.reduce(
            meta_graphs, weights=weights, progress_bar=progress_bar, processes=processes, **self.meta_graph_config
        )
        return meta_graph