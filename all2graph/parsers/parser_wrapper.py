import os
from typing import Tuple, List, Dict, Union

import pandas as pd

from .data_parser import DataParser
from .graph_parser import GraphParser
from .post_parser import PostParser
from ..graph import Graph, RawGraph
from ..meta_struct import MetaStruct
from ..utils import iter_csv, mp_run


UnionGraphParser = Union[GraphParser, Tuple[GraphParser, List[str]]]
UnionPostParser = Union[PostParser, Tuple[PostParser, List[str]]]


class ParserWrapper(MetaStruct):
    def __init__(
            self,
            data_parser: Union[DataParser, List[DataParser], Dict[str, DataParser]] = None,
            graph_parser: Union[UnionGraphParser, List[UnionGraphParser], Dict[str, UnionGraphParser]] = None,
            post_parser: Union[UnionPostParser, List[UnionPostParser], Dict[str, UnionPostParser]] = None
    ):
        super().__init__(initialized=True)
        self.data_parser = data_parser
        self.graph_parser = graph_parser
        self.post_parser = post_parser

    @property
    def data_parser(self):
        if len(self._data_parser) == 1:
            return list(self._data_parser.values())[0]
        return self._data_parser

    @data_parser.setter
    def data_parser(self, data_parser):
        if not isinstance(data_parser, (list, dict)):
            data_parser = [data_parser]
        if isinstance(data_parser, list):
            data_parser = {i: parser for i, parser in enumerate(data_parser)}
        self._data_parser: Dict[str, DataParser] = data_parser

    @property
    def graph_parser(self):
        if len(self._graph_parser) == 1:
            return list(self._graph_parser.values())[0][0]
        return {k: v[0] for k, v in self.graph_parser.items()}

    @graph_parser.setter
    def graph_parser(self, graph_parser):
        if not isinstance(graph_parser, (list, dict)):
            graph_parser = [graph_parser]
        if isinstance(graph_parser, list):
            graph_parser = {i: parser for i, parser in enumerate(graph_parser)}
        self._graph_parser: Dict[str, Tuple[GraphParser, List[str]]] = {}
        for k, parser in graph_parser.items():
            if not isinstance(parser, tuple):
                parser = (parser, list(self._data_parser))
                self._graph_parser[k] = parser

    @property
    def post_parser(self):
        if len(self._post_parser) == 1:
            return list(self._post_parser.values())[0][0]
        return {k: v[0] for k, v in self._post_parser.items()}

    @post_parser.setter
    def post_parser(self, post_parser):
        if post_parser is None:
            self._post_parser = None
        else:
            if not isinstance(post_parser, (list, dict)):
                post_parser = [post_parser]
            if isinstance(post_parser, list):
                post_parser = {i: parser for i, parser in enumerate(post_parser)}
            self._post_parser: Dict[str, Tuple[PostParser, List[str]]] = {}
            for k, parser in post_parser.items():
                if not isinstance(parser, tuple):
                    parser = (parser, list(self._graph_parser))
                self._post_parser[k] = parser

    def call_post_parser(self, graph: Graph, key: str = None) -> Dict[Tuple, Graph]:
        if self._post_parser is None:
            return {(key, ): graph}
        output = {}
        for key2, (parser, key3) in self._post_parser.items():
            if key and key not in key3:
                continue
            output[(key, key2)] = parser(graph)
        return output

    def call_graph_parser(self, raw_graph: RawGraph, key: str = None, post=True) -> Dict[Tuple, Graph]:
        output = {}
        for key2, (parser, key3) in self._graph_parser.items():
            if key and key not in key3:
                continue
            if post:
                for kk, graph in self.call_post_parser(parser(raw_graph), key=key2).items():
                    output[(key, ) + kk] = graph
            else:
                output[(key, key2, )] = parser(raw_graph)
        return output

    def __call__(self, df: pd.DataFrame, disable=True, return_df=False, sel_cols=None, drop_cols=None, post=True
                 ) -> Union[Union[Graph, dict], Tuple[Union[Graph, dict], pd.DataFrame]]:
        output = {}
        for k, parser in self._data_parser.items():
            raw_graph = parser(df, disable=disable)
            output.update(self.call_graph_parser(raw_graph, key=k, post=post))
        if len(output) == 1:
            output = list(output.values())[0]
        if return_df:
            if sel_cols is not None:
                df = df[sel_cols]
            if drop_cols is not None:
                df = df.drop(columns=drop_cols)
            return output, df.drop(columns=[parser.data_col for parser in self._data_parser.values()])
        else:
            return output

    def labels(self, df):
        labels = {}
        for data_key, data_parser in self._data_parser.items():
            labels.update(data_parser.get_targets(df))
        return labels

    def _save(self, inputs, **kwargs):
        """
        不执行post parser
        Args:
            inputs:
            **kwargs: __call__的额外参数

        Returns:

        """
        df, path = inputs
        graph, df = self(df, return_df=True, **kwargs)
        labels = self.labels(df)
        graph.save(path, labels=labels)
        df = df.copy()
        df['path'] = path
        return df

    def save(self, src, dst, disable=False, chunksize=64, postfix='saving graph', processes=None, sel_cols=None,
             drop_cols=None, post=False, **kwargs):
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
            post: post parser是否生效
            **kwargs: pd.read_csv的额外参数

        Returns:
            返回一个包含路径和元数据的DataFrame
        """
        assert not os.path.exists(dst), '{} already exists'.format(dst)
        os.mkdir(dst)
        dfs = []
        inputs = (
            (df, os.path.join(dst, '{}.ag.graph'.format(i)))
            for i, df in enumerate(iter_csv(src, chunksize=chunksize, **kwargs))
        )
        kwds = dict(sel_cols=sel_cols, drop_cols=drop_cols, post=post)
        for df in mp_run(self._save, inputs, kwds=kwds, disable=disable, processes=processes, postfix=postfix):
            dfs.append(df)
        path_df = pd.concat(dfs)
        path_df.to_csv(os.path.abspath(dst) + '_path.csv', index=False)
        return pd.concat(dfs)

    def generator(
            self, src, disable=False, chunksize=64, processes=None, postfix='parsing', return_df=False, sel_cols=None,
            drop_cols=None, **kwargs):
        """
        返回一个graph生成器
        Args:
            src: dataframe 或者"list和路径的任意嵌套"
            disable: 是否禁用进度条
            chunksize: 批次处理的大小
            processes: 多进程数量
            postfix: 进度条后缀
            return_df: 是否返回graph之外的东西
            sel_cols: 需要返回的元数据列
            drop_cols: 需要去掉的列，只在meta_col为None时生效
            **kwargs: pd.read_csv的额外参数

        Returns:
            graph:
            df
        """
        data = iter_csv(src, chunksize=chunksize, **kwargs)
        kwds = dict(sel_cols=sel_cols, drop_cols=drop_cols)
        for graph, df in mp_run(self, data, kwds=kwds, processes=processes, disable=disable, postfix=postfix):
            if return_df:
                yield graph, df
            else:
                yield graph

    def __eq__(self, other):
        return self.data_parser == other.data_parser and self.graph_parser == other.graph_parser\
               and self.post_parser == other.post_parser
