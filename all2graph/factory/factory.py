import sys
from typing import Callable, Iterable, Tuple

import pandas as pd
import toad
from toad.utils.progress import Progress

from ..utils import dataframe_chunk_iter
from ..meta_graph import MetaGraph
from ..resolver import Resolver
from ..transformer import Transformer


class Factory:
    def __init__(self, resolver: Resolver, preprocessor: Callable[[pd.DataFrame], Iterable], **kwargs):
        self.preprocessor = preprocessor
        self.resolver = resolver
        self.transformer_config = kwargs

    def produce(
            self, data: Iterable[pd.DataFrame], progress_bar=False, chunksize=None, df_kwgs=None, suffix='reading csv',
            **kwargs
    ) -> Tuple[MetaGraph, Transformer]:
        """

        :param data:
        :param progress_bar:
        :param chunksize:
        :param df_kwgs:
        :param suffix:
        :param kwargs:
        :return:
        """
        df_kwgs = df_kwgs or {}
        if isinstance(data, (str, pd.DataFrame)):
            data = dataframe_chunk_iter(data, chunksize=chunksize, **df_kwgs)
        if progress_bar and not isinstance(data, Progress):
            data = Progress(data)
            if toad.version.__version__ <= '0.0.65' and data.size is None:
                data.size = sys.maxsize
            data.suffix = suffix

        meta_graphs = []
        weights = []
        for chunk in data:
            weights.append(chunk.shape[0])
            data = self.preprocessor(chunk)
            graph, global_index_mapper, local_index_mappers = self.resolver.resolve(data, progress_bar=False)
            index_ids = list(global_index_mapper.values())
            for mapper in local_index_mappers:
                index_ids += list(mapper.values())
            meta_graphs.append(MetaGraph.from_data(graph, index_nodes=index_ids, progress_bar=False, **kwargs))

        meta_graph = MetaGraph.reduce(meta_graphs, weights=weights, progress_bar=progress_bar, **kwargs)
        transformer = Transformer.from_meta_graph(meta_graph, **self.transformer_config)
        return meta_graph, transformer
