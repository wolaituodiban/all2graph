import os.path
from multiprocessing import Pool
from typing import Iterable, Tuple, List, Union

import pandas as pd


from ..graph import RawGraph
from ..meta import MetaInfo
from ..parsers import DataParser, RawGraphParser
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
                graph: 输入，需要是RawGraph
                index_nodes: index node的坐标，用于排除index
                disable: 禁用进度条
                num_bins: 统计各种分布时的分箱数量
            raw_graph_parser_config:
                meta_info:
                min_df: 字符串最小文档平吕
                max_df: 字符串最大文档频率
                top_k: 选择前k个字符串
                top_method: 'max_tfidf', 'mean_tfidf', 'max_tf', 'mean_tf', 'max_tc', mean_tc'
                targets:
                tokenizer:
                filter_key:

        """

        super().__init__(initialized=True)
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

    def enable_preprocessing(self):
        self.data_parser.enable_preprocessing()

    def disable_preprocessing(self):
        self.data_parser.disable_preprocessing()

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
                results = pool.imap(self._analyse, data)
                results = tqdm(results, disable=disable, postfix=postfix)
                for meta_info, weight in results:
                    meta_infos.append(meta_info)
                    weights.append(weight)

        meta_info = MetaInfo.reduce(
            meta_infos, weights=weights, disable=disable, processes=processes, **self.meta_info_config
        )
        self.raw_graph_parser = RawGraphParser.from_data(meta_info, **self.graph_parser_config)
        return meta_info

    def _save(self, x: Tuple[pd.DataFrame, str, bool, bool, Union[None, list]]) -> pd.DataFrame:
        """
        将原始数据加工后保存成一个文件
        Args:
            x: 包含五个变量
                df: DataFrame，原始数据
                dst: 文件夹路径
                zip: 是否压缩，仅在raw=True是生效
                raw: 是否保存成RawGraph
                meta_col: 需要返回的元数据列

        Returns:
            返回一个包含路径和元数据的DataFrame
        """
        df, dst, zip, raw, meta_col = x
        if raw:
            path = '.'.join([dst, 'zip' if zip else 'csv'])
            self.data_parser.save(df, path)
        else:
            path = '.'.join([dst, 'all2graph.graph'])
            raw_graph, *_ = self.data_parser.parse(df, disable=True)
            graph = self.raw_graph_parser.parse(raw_graph)
            labels = self.data_parser.gen_targets(df, self.targets)
            graph.save(path, labels=labels)

        meta_df = pd.DataFrame({'path': [path] * df.shape[0]})
        if meta_col is not None:
            meta_df[meta_col] = df[meta_col].values
        return meta_df

    def save(
            self, src, dst, disable=False, zip=True, error=True, warning=True, concat_chip=True, chunksize=64,
            postfix=None, processes=None, raw=False, meta_col=None, **kwargs):
        """
        将原始数据加工后，存储成分片的文件
        Args:
            src: 原始数据
            dst: 存储文件夹路径
            disable: 是否禁用进度条
            zip: 是否压缩，仅在raw=True是生效
            error: 读取数据遇到错误时报错
            warning: 读取数据遇到错误是报警
            concat_chip: 合并样本，强制保证每个分片的大小都是chunksize
            chunksize: 分片数据的大小
            postfix: 进度条后缀
            processes: 多进程数量
            raw: 是否保存成RawGraph
            meta_col: 需要返回的元数据列
            **kwargs: 传递给dataframe_chunk_iter的额外参数

        Returns:
            返回一个包含路径和元数据的DataFrame
        """
        # assert meta_col is None or isinstance(meta_col, list)
        assert not os.path.exists(dst), '{} already exists'.format(dst)
        if postfix is None:
            postfix = 'saving{}graph'.format(' raw ' if raw else ' ')
        os.mkdir(dst)
        generator = dataframe_chunk_iter(
                src, error=error, warning=warning, concat_chip=concat_chip, chunksize=chunksize, **kwargs)
        generator = ((df, os.path.join(dst, str(i)), zip, raw, meta_col) for i, df in enumerate(generator))
        generator = tqdm(generator, disable=disable, postfix=postfix)
        if processes == 0:
            meta_df = pd.concat(map(self._save, generator))
        else:
            with Pool(processes) as pool:
                meta_df = pd.concat(pool.imap(self._save, generator))
        return meta_df

    def produce_graph_and_label(self, chunk: pd.DataFrame):
        graph, *_ = self._produce_raw_graph(chunk)
        x = self.raw_graph_parser.parse(graph)
        labels = self.data_parser.gen_targets(chunk, target_cols=self.targets)
        return x, labels

    def produce_dataloader(
            self, src, dst=None, disable=False, zip=True, error=True, warning=True, concat_chip=True, chunksize=64,
            shuffle=True, csv_configs=None, raw_graph=False, graph=False, processes=None, v2=False, meta_df=None,
            num_workers=0, **kwargs):
        """

        Args:
            src:
            dst: 如果不是None，src中的数据，将被分片存储在dst中
            disable:
            zip:
            error:
            warning:
            concat_chip:
            chunksize:
            shuffle:
            csv_configs:
            raw_graph: 如果True，那么数据源视为RawGraph
            graph: 如果True，那么数据源视为Graph，并且覆盖raw_graph的效果
            processes: 当dst不是None时，执行save方法时，多进程的个数
            v2: 使用dataset v2版本
            meta_df: 返回一个包含路径和元数据的DataFrame，需要有一列path，如果提供了dst，按么会使用分片存储后的meta_df
            num_workers: DataLoader多进程数量
            **kwargs: DataLoader的额外参数

        Returns:

        """
        from torch.utils.data import DataLoader

        if graph:
            from ..data import GraphDataset

            if dst is not None:
                # 存储Graph文件
                self.save(
                    src, dst, disable=disable, zip=zip, error=error, warning=warning, concat_chip=concat_chip,
                    chunksize=chunksize, raw=False, processes=processes, **(csv_configs or {}))
                src = dst

            dataset = GraphDataset(src)
            return DataLoader(dataset, batch_size=None, sampler=None, **kwargs)
        else:
            if dst is not None:
                if raw_graph:
                    # 存储RawGraph csv文件
                    meta_df = self.save(
                        src, dst, disable=disable, zip=zip, error=error, warning=warning, concat_chip=concat_chip,
                        chunksize=chunksize, raw=True, processes=processes, **(csv_configs or {}))
                    self.disable_preprocessing()  # 改变data_parser的模式
                else:
                    # 分割csv
                    meta_df = split_csv(
                        src=src, dst=dst, chunksize=chunksize, disable=disable, zip=zip, error=error, warning=warning,
                        concat_chip=concat_chip, **(csv_configs or {}))
                    self.enable_preprocessing()
                src = dst

            if v2:
                # 使用v2版本的dataset
                from ..data import CSVDatasetV2
                dataset = CSVDatasetV2(
                    src=meta_df, data_parser=self.data_parser, raw_graph_parser=self.raw_graph_parser,
                    **(csv_configs or {}))
                sampler = dataset.build_sampler(shuffle=shuffle, num_workers=num_workers)
                return DataLoader(dataset, collate_fn=dataset.collate_fn, sampler=sampler, **kwargs)
            else:
                # 使用老版本的dataset
                from ..data import CSVDataset
                dataset = CSVDataset(
                    src, data_parser=self.data_parser, raw_graph_parser=self.raw_graph_parser, chunksize=chunksize,
                    shuffle=shuffle, disable=disable, error=error, warning=warning, **(csv_configs or {}))
                return DataLoader(
                    dataset, shuffle=shuffle, collate_fn=dataset.collate_fn, num_workers=num_workers, **kwargs)

    def produce_model(
            self, d_model: int, nhead: int, num_layers: List[int], encoder_configs=None, learner_configs=None,
            mock=False):
        from ..nn import Encoder, EncoderMetaLearner, EncoderMetaLearnerMocker
        encoder = Encoder(
            num_embeddings=self.raw_graph_parser.num_strings, d_model=d_model, nhead=nhead,
            num_layers=num_layers, **(encoder_configs or {}))
        if mock:
            model = EncoderMetaLearnerMocker(raw_graph_parser=self.raw_graph_parser, encoder=encoder)
        else:
            model = EncoderMetaLearner(
                raw_graph_parser=self.raw_graph_parser, encoder=encoder, **(learner_configs or {}))
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
