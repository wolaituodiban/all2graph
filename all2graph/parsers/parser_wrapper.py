import os
from typing import Tuple, List, Dict, Union

import pandas as pd

from .data_parser import DataParser
from .graph_parser import GraphParser
from ..graph import Graph
from ..meta_struct import MetaStruct
from ..utils import iter_csv, mp_run


UnionType = Union[GraphParser, Tuple[GraphParser, List[str]]]


class ParserWrapper(MetaStruct):
    def __init__(
            self,
            data_parser: Union[DataParser, List[DataParser], Dict[str, DataParser]],
            graph_parser: Union[UnionType, List[UnionType], Dict[str, UnionType]]
    ):
        super().__init__(initialized=True)
        if not isinstance(data_parser, (list, dict)):
            data_parser = [data_parser]
        if isinstance(data_parser, list):
            data_parser = {i: parser for i, parser in enumerate(data_parser)}
        self.data_parser: Dict[str, DataParser] = data_parser

        if not isinstance(graph_parser, (list, dict)):
            graph_parser = [graph_parser]
        if isinstance(graph_parser, list):
            graph_parser = {i: parser for i, parser in enumerate(graph_parser)}
        self.graph_parser: Dict[str, Tuple[GraphParser, List[str]]] = {}
        for k, parser in graph_parser.items():
            if not isinstance(parser, tuple):
                parser = (parser, list(self.data_parser))
            self.graph_parser[k] = parser

    def __call__(self, df: pd.DataFrame, disable=True, return_df=True) -> Union[Graph, Tuple[Graph, pd.DataFrame]]:
        raw_graph = {k: parser(df, disable=disable) for k, parser in self.data_parser.items()}
        graph = {}
        for graph_key, (graph_parser, data_keys) in self.graph_parser.items():
            for data_key in data_keys:
                out_key = '{}_{}'.format(data_key, graph_key)
                graph[out_key] = graph_parser(raw_graph[data_key])
        if len(graph) == 1:
            graph = list(graph.values())[0]
        if return_df:
            return graph, df.drop(columns=[parser.json_col for parser in self.data_parser.values()])
        else:
            return graph

    def labels(self, df):
        labels = {}
        for data_key, data_parser in self.data_parser.items():
            labels.update(data_parser.get_targets(df))
        return labels

    def graph_and_labels(self, df):
        return self(df, return_df=False), self.labels(df)

    def generator(
            self, src, disable=False, chunksize=64, processes=None, postfix='parsing', return_df=False, **kwargs):
        """
        返回一个graph生成器
        Args:
            src: dataframe 或者"list和路径的任意嵌套"
            disable: 是否禁用进度条
            chunksize: 批次处理的大小
            processes: 多进程数量
            postfix: 进度条后缀
            return_df: 是否返回graph之外的东西
            **kwargs: pd.read_csv的额外参数

        Returns:
            graph:
            df
        """
        data = iter_csv(src, chunksize=chunksize, **kwargs)
        for graph, df in mp_run(self, data, processes=processes, disable=disable, postfix=postfix):
            if return_df:
                yield graph, df
            else:
                yield graph

    def save(
            self, src, dst, disable=False, chunksize=64,
            postfix='saving graph', processes=None, sel_cols=None, drop_cols=None, **kwargs):
        """
        将原始数据加工后，存储成分片的文件
        Args:
            src: 原始数据
            dst: 存储文件夹路径
            disable: 是否禁用进度条
            chunksize: 分片数据的大小
            postfix: 进度条后缀
            processes: 多进程数量
            sel_cols: 需要返回的元数据列
            drop_cols: 需要去掉的列，只在meta_col为None时生效
            **kwargs: pd.read_csv的额外参数

        Returns:
            返回一个包含路径和元数据的DataFrame
        """
        # assert meta_col is None or isinstance(meta_col, list)
        assert not os.path.exists(dst), '{} already exists'.format(dst)
        os.mkdir(dst)
        dfs = []
        for i, (graph, df) in enumerate(
                self.generator(
                    src, disable=disable, chunksize=chunksize, postfix=postfix, processes=processes, return_df=True,
                    **kwargs)):
            if sel_cols is not None:
                df = df[sel_cols]
            if drop_cols is not None:
                df = df.drop(columns=drop_cols)

            labels = self.labels(df)
            if isinstance(graph, dict):
                for k, g in graph.items():
                    path = os.path.join(dst, '{}_{}.all2graph.graph'.format(i, k))
                    g.save(path, labels=labels)
                    df['path_{}'.format(k)] = path
            else:
                path = os.path.join(dst, '{}.all2graph.graph'.format(i))
                graph.save(path, labels=labels)
                df['path'] = path
            dfs.append(df)
        path_df = pd.concat(dfs)
        path_df.to_csv(dst + '_path.csv', index=False)
        return pd.concat(dfs)

    def __eq__(self, other):
        return self.data_parser == other.data_parser and self.graph_parser == other.data_parser

    @classmethod
    def from_data(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_json(cls, obj: dict):
        raise NotImplementedError

    def to_json(self) -> dict:
        raise NotImplementedError

    @classmethod
    def reduce(cls, structs, weights=None, **kwargs):
        raise NotImplementedError
