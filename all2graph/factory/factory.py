import os
from multiprocessing import Pool
from typing import Iterable, Tuple, List, Union

import pandas as pd
from torch.utils.data import DataLoader

from ..dataset import GraphDataset
from ..graph import RawGraph
from ..meta import MetaInfo
from ..parsers import DataParser, RawGraphParser
from ..utils import progress_wrapper
from ..utils.pd_utils import dataframe_chunk_iter, split_csv
from ..meta_struct import MetaStruct


def _verify_kwargs(func, *args, kwargs):
    unexpected = []
    for k, v in kwargs.items():
        try:
            func(*args, **{k: v})
        except TypeError:
            unexpected.append(k)
        except:
            pass
    return unexpected


class Factory(MetaStruct):
    def __init__(
            self, data_parser: DataParser, meta_info_config: dict = None, raw_graph_parser_config: dict = None,
            **kwargs):
        super().__init__(initialized=True, **kwargs)
        if meta_info_config is not None:
            unexp_args = _verify_kwargs(MetaInfo.from_data, RawGraph(), kwargs=meta_info_config)
            assert len(unexp_args) == 0, 'meta_info_config got unexpected keyword argument {}'.format(unexp_args)
        if raw_graph_parser_config is not None:
            unexp_args = _verify_kwargs(RawGraphParser.from_data, None, kwargs=raw_graph_parser_config)
            assert len(unexp_args) == 0, 'raw_graph_parser_config got unexpected keyword argument {}'.format(unexp_args)

        self.data_parser = data_parser
        self.meta_info_config = meta_info_config or {}
        self.graph_parser_config = raw_graph_parser_config or {}
        self.raw_graph_parser: Union[RawGraphParser, None] = None
        self.save_path = None  # 多进程的cache

    @property
    def targets(self):
        if self.raw_graph_parser is None:
            return []
        else:
            return self.raw_graph_parser.targets

    def _produce_raw_graph(self, chunk):
        return self.data_parser.parse(chunk, progress_bar=False)

    def _analyse(self, chunk: pd.DataFrame) -> Tuple[MetaInfo, int]:
        graph, global_index_mapper, local_index_mappers = self._produce_raw_graph(chunk)
        index_ids = list(global_index_mapper.values())
        for mapper in local_index_mappers:
            index_ids += list(mapper.values())
        meta_info = MetaInfo.from_data(graph, index_nodes=index_ids, progress_bar=False, **self.meta_info_config)
        return meta_info, chunk.shape[0]

    def analyse(self, data: Union[pd.DataFrame, Iterable[pd.DataFrame]], chunksize=64, progress_bar=True,
                postfix='reading csv', processes=None, **kwargs) -> MetaInfo:
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
            meta_infos, weights=weights, progress_bar=progress_bar, processes=processes, **self.meta_info_config
        )
        self.raw_graph_parser = RawGraphParser.from_data(meta_info, **self.graph_parser_config)
        return meta_info

    def produce_graph_and_label(self, chunk: pd.DataFrame):
        graph, *_ = self._produce_raw_graph(chunk)
        x = self.raw_graph_parser.parse(graph)
        labels = self.data_parser.gen_targets(chunk, target_cols=self.targets)
        return x, labels

    def produce_dataloader(
            self, path: Union[str, List[str]], batch_size, num_workers=0, pin_memory=False, partitions=1, disable=False,
            **kwargs) -> DataLoader:
        dir_path = path
        if isinstance(path, str) and not os.path.isdir(path):
            if dir_path.endswith('.csv') or dir_path.endswith('.zip'):
                dir_path = dir_path[:-4]
            os.mkdir(dir_path)
            print('dir_path: {}'.format(dir_path))
            split_csv(path, dir_path, chunksize=batch_size, disable=disable, zip=True, **kwargs)
        dataset = GraphDataset(dir_path, factory=self, partitions=partitions, shuffle=True, disable=disable, **kwargs)
        return DataLoader(
            dataset, batch_size=partitions, shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
            collate_fn=dataset.collate_fn
        )

    def extra_repr(self) -> str:
        return 'data_parser: {}\ngraph_parser: {}'.format(
            self.data_parser, self.raw_graph_parser
        )

    def __eq__(self, other):
        return super().__eq__(other) \
               and self.raw_graph_parser == other.raw_graph_parser and self.data_parser == other.raw_graph_parser

    @classmethod
    def from_data(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_json(cls, obj: dict):
        raise NotImplementedError

    @classmethod
    def reduce(cls, structs, weights=None, **kwargs):
        raise NotImplementedError

    def to_json(self) -> dict:
        raise NotImplementedError
