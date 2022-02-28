import os.path
from multiprocessing import Pool
from typing import Iterable, Tuple, List, Union

import pandas as pd


from ..graph import RawGraph
from ..info import MetaInfo
from ..parsers import DataParser, GraphParser
from ..utils import tqdm
from ..utils.file_utils import dataframe_chunk_iter, split_csv
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
    """Factory"""
    def __init__(
            self, data_parser: DataParser, meta_info_config: dict = None, raw_graph_parser_config: dict = None):
        """

        Args:
            data_parser:
            meta_info_config:
                num_bins: 统计各种分布时的分箱数量
            raw_graph_parser_config:
                min_df: 字符串最小文档频率
                max_df: 字符串最大文档频率
                top_k: 选择前k个字符串
                top_method: 'max_tfidf', 'mean_tfidf', 'max_tf', 'mean_tf', 'max_tc', mean_tc'
                targets:
                tokenizer:
                filter_key:
                scale_method: 'prob' or 'minmax_scale'
                scale_kwargs: 具体见MetaNumber.get_probs和MetaNumber.minmax_scale
        """

        super().__init__(initialized=True)
        if meta_info_config is not None:
            unexp_args = _verify_kwargs(MetaInfo.from_data, RawGraph(), kwargs=meta_info_config)
            assert len(unexp_args) == 0, 'meta_info_config got unexpected keyword argument {}'.format(unexp_args)
        if raw_graph_parser_config is not None:
            unexp_args = _verify_kwargs(GraphParser.from_data, None, kwargs=raw_graph_parser_config)
            assert len(unexp_args) == 0, 'raw_graph_parser_config got unexpected keyword argument {}'.format(unexp_args)

        self.data_parser = data_parser
        self.meta_info_config = meta_info_config or {}
        self.raw_graph_parser_config = raw_graph_parser_config or {}
        self.raw_graph_parser: Union[GraphParser, None] = None
        self.save_path = None  # 多进程的cache

    @property
    def targets(self):
        if self.raw_graph_parser is None:
            return []
        else:
            return self.raw_graph_parser.targets

    def _produce_raw_graph(self, chunk):
        return self.data_parser.parse(chunk, disable=True)

    def _analyse(self, chunk: pd.DataFrame) -> Tuple[MetaInfo, int]:
        graph, global_index_mapper, local_index_mappers = self._produce_raw_graph(chunk)
        index_ids = list(global_index_mapper.values())
        for mapper in local_index_mappers:
            index_ids += list(mapper.values())
        meta_info = MetaInfo.from_data(graph, index_nodes=index_ids, disable=True, **self.meta_info_config)
        return meta_info, chunk.shape[0]

    def analyse(self, data: Union[pd.DataFrame, Iterable[pd.DataFrame]], chunksize=64, disable=False,
                postfix='reading csv', processes=None, **kwargs) -> MetaInfo:
        data = dataframe_chunk_iter(data, chunksize=chunksize, **kwargs)
        meta_infos: List[MetaInfo] = []
        weights = []
        if processes == 0:
            results = map(self._analyse, data)
            results = tqdm(results, disable=disable, postfix=postfix)
            for meta_info, weight in results:
                meta_infos.append(meta_info)
                weights.append(weight)
        else:
            with Pool(processes) as pool:
                results = pool.imap_unordered(self._analyse, data)
                results = tqdm(results, disable=disable, postfix=postfix)
                for meta_info, weight in results:
                    meta_infos.append(meta_info)
                    weights.append(weight)

        meta_info = MetaInfo.reduce(
            meta_infos, weights=weights, disable=disable, processes=processes, **self.meta_info_config
        )
        self.raw_graph_parser = GraphParser.from_data(meta_info, **self.raw_graph_parser_config)
        return meta_info

    def produce_graph_and_label(self, chunk: pd.DataFrame):
        graph, *_ = self._produce_raw_graph(chunk)
        x = self.raw_graph_parser.parse(graph)
        labels = self.data_parser.gen_targets(chunk, target_cols=self.targets)
        return x, labels

    def _save(self, x: Tuple[pd.DataFrame, str, Union[None, list], Union[None, list]]) -> pd.DataFrame:
        """
        将原始数据加工后保存成一个文件
        Args:
            x: 包含五个变量
                df: DataFrame，原始数据
                dst: 文件夹路径
                meta_col: 需要返回的元数据列

        Returns:
            返回一个包含路径和元数据的DataFrame
        """
        df, dst, meta_col, drop_col = x
        path = '.'.join([dst, 'all2graph.graph'])
        raw_graph, *_ = self.data_parser.parse(df, disable=True)
        graph, labels = self.produce_graph_and_label(df)
        graph.save(path, labels=labels)

        if meta_col is not None:
            meta_df = df[meta_col]
            meta_df['path'] = path
        elif drop_col is not None:
            meta_df = df.drop(columns=drop_col)
            meta_df['path'] = path
        else:
            meta_df = pd.DataFrame({'path': [path] * df.shape[0]})
        return meta_df

    def save(
            self, src, dst, disable=False, chunksize=64,
            postfix=None, processes=None, meta_cols=None, drop_cols=None, **kwargs):
        """
        将原始数据加工后，存储成分片的文件
        Args:
            src: 原始数据
            dst: 存储文件夹路径
            disable: 是否禁用进度条
            chunksize: 分片数据的大小
            postfix: 进度条后缀
            processes: 多进程数量
            meta_cols: 需要返回的元数据列
            drop_cols: 需要去掉的列，只在meta_col为None时生效
            **kwargs: 传递给dataframe_chunk_iter的额外参数

        Returns:
            返回一个包含路径和元数据的DataFrame
        """
        # assert meta_col is None or isinstance(meta_col, list)
        assert not os.path.exists(dst), '{} already exists'.format(dst)
        if postfix is None:
            postfix = 'saving graph'
        os.mkdir(dst)
        generator = dataframe_chunk_iter(src, chunksize=chunksize, **kwargs)
        generator = ((df, os.path.join(dst, str(i)), meta_cols, drop_cols) for i, df in enumerate(generator))
        if processes == 0:
            meta_df = pd.concat(tqdm(map(self._save, generator), disable=disable, postfix=postfix))
        else:
            with Pool(processes) as pool:
                meta_df = pd.concat(tqdm(pool.imap(self._save, generator), disable=disable, postfix=postfix))
        meta_df.to_csv(dst+'_meta.zip', index=False)
        return meta_df

    def produce_dataloader(
            self, df=None, shuffle=True, csv_configs=None, graph=False, meta_df=None,
            num_workers=0, batch_size=1, **kwargs):
        """

        Args:
            df: path or list of path
            shuffle:
            csv_configs:
            graph: 如果True，那么数据源视为Graph
            meta_df: 返回一个包含路径和元数据的DataFrame，需要有一列path，如果提供了dst，按么会使用分片存储后的meta_df
            num_workers: DataLoader多进程数量
            batch_size: batch大小
            **kwargs: DataLoader的额外参数

        Returns:

        """
        if df is None:
            if graph:
                from ..data import GraphDataset
                dataset = GraphDataset(src=meta_df)
            else:
                from ..data import CSVDatasetV2
                dataset = CSVDatasetV2(
                    src=meta_df, data_parser=self.data_parser, raw_graph_parser=self.raw_graph_parser,
                    **(csv_configs or {}))
        else:
            from ..data import DFDataset
            dataset = DFDataset(
                df=df, data_parser=self.data_parser, raw_graph_parser=self.raw_graph_parser
            )
        return dataset.build_dataloader(num_workers=num_workers, shuffle=shuffle, batch_size=batch_size, **kwargs)

    def produce_model(
            self, d_model: int, nhead: int, num_layers: List[int], encoder_config=None, learner_config=None,
            mock=True):
        from ..nn import Encoder, EncoderMetaLearner, EncoderMetaLearnerMocker
        encoder = Encoder(
            num_embeddings=self.raw_graph_parser.num_strings, d_model=d_model, nhead=nhead,
            num_layers=num_layers, **(encoder_config or {}))
        if mock:
            model = EncoderMetaLearnerMocker(raw_graph_parser=self.raw_graph_parser, encoder=encoder)
        else:
            model = EncoderMetaLearner(
                raw_graph_parser=self.raw_graph_parser, encoder=encoder, **(learner_config or {}))
        return model

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

    def set_filter_key(self, x):
        self.raw_graph_parser.set_filter_key(x)
