from multiprocessing import Pool
from typing import Iterable, Tuple, List, Union

import pandas as pd

from ..meta import MetaInfo
from ..data import DataParser
from ..graph.parser import GraphParser
from ..utils import progress_wrapper
from ..utils.pd_utils import dataframe_chunk_iter


class Factory:
    def __init__(self, data_parser: DataParser,
                 meta_graph_config: dict = None, graph_transer_config: dict = None):
        self.data_parser = data_parser
        self.meta_config = meta_graph_config or {}
        self.graph_parser_config = graph_transer_config or {}
        self.graph_parser: Union[GraphParser, None] = None
        self.save_path = None  # 多进程的cache

    @property
    def targets(self):
        if self.graph_parser is None:
            return []
        else:
            return self.graph_parser.targets

    def _produce_graph(self, chunk):
        return self.data_parser.parse(chunk, progress_bar=False, targets=self.targets)

    def _analyse(self, chunk: pd.DataFrame) -> Tuple[MetaInfo, int]:
        graph, global_index_mapper, local_index_mappers = self._produce_graph(chunk)
        index_ids = list(global_index_mapper.values())
        for mapper in local_index_mappers:
            index_ids += list(mapper.values())
        meta_info = MetaInfo.from_data(graph, index_nodes=index_ids, progress_bar=False, **self.meta_config)
        return meta_info, chunk.shape[0]

    def analyse(self, data: Union[pd.DataFrame, Iterable[pd.DataFrame]], chunksize=64, progress_bar=False,
                postfix='reading csv', processes=0, **kwargs) -> MetaInfo:
        if isinstance(data, (str, pd.DataFrame)):
            data = dataframe_chunk_iter(data, chunksize=chunksize, **kwargs)

        meta_infos: List[MetaInfo] = []
        weights = []
        if processes == 0:
            results = map(self._analyse, data)
            results = progress_wrapper(results, disable=not progress_bar, postfix=postfix)
            for meta_info, weight in results:
                meta_infos.append(meta_info)
                weights.append(weight)
        else:
            with Pool(processes) as pool:
                results = pool.imap(self._analyse, data)
                results = progress_wrapper(results, disable=not progress_bar, postfix=postfix)
                for meta_info, weight in results:
                    meta_infos.append(meta_info)
                    weights.append(weight)

        meta_info = MetaInfo.reduce(
            meta_infos, weights=weights, progress_bar=progress_bar, processes=processes, **self.meta_config
        )
        self.graph_parser = GraphParser.from_data(meta_info, **self.graph_parser_config)
        return meta_info

    def produce_dgl_graph_and_label(self, chunk: pd.DataFrame):
        graph, *_ = self._produce_graph(chunk)
        dgl_meta_graph, dgl_graph = self.graph_parser.graph_to_dgl(graph)
        labels = self.data_parser.gen_targets(chunk, target_cols=self.targets)
        return (dgl_meta_graph, dgl_graph), labels
