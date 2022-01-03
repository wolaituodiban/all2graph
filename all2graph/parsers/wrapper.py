import time
from typing import Tuple, List, Dict, Union

import pandas as pd

from .data import DataParser
from .graph import RawGraphParser
from ..utils import dataframe_chunk_iter, tqdm


class ParserWrapper:
    def __init__(
            self, data_parser: Union[DataParser, Dict[str, DataParser]], raw_graph_parser: RawGraphParser,
            temp_file=False):
        if isinstance(data_parser, DataParser):
            self.data_parsers = {'output': data_parser}
        elif isinstance(data_parser, dict):
            self.data_parsers = dict(data_parser)
        else:
            raise TypeError('data_parser must be DataParse or dict')
        self.raw_graph_parser = raw_graph_parser
        self.temp_file = temp_file

    @property
    def json_cols(self):
        return [parser.json_col for parser in self.data_parsers.values()]

    def parse(
            self, df: pd.DataFrame, disable=True
    ) -> Tuple[pd.DataFrame, dict, Dict[str, dict], Dict[str, List[dict]]]:
        """

        Args:
            df:
            disable:

        Returns:
            df: df which drop json cols
            graphs: dict of all2graph.Graph or str if temp_file == True
            global_index_mappers:
            local_index_mapperss
        """
        graphs = {}
        global_index_mappers = {}
        local_index_mapperss = {}
        for name, parser in self.data_parsers.items():
            raw_graph, global_index_mapper, local_index_mappers = parser.parse(df, disable=disable)
            graph = self.raw_graph_parser.parse(raw_graph)
            if self.temp_file:
                filename = str(time.time())
                filename = '+'.join([filename, str(id(filename))])+'.all2graph.graph'
                graph.save(filename)
                graph = filename
            graphs[name] = graph
            global_index_mappers[name] = global_index_mapper
            local_index_mapperss[name] = local_index_mappers
        return df.drop(columns=self.json_cols), graphs, global_index_mappers, local_index_mapperss

    def generate(
            self, src, disable=False, chunksize=64, processes=0, postfix='parsing', graph_only=False, mp='auto',
            **kwargs):
        """
        返回一个graph生成器
        Args:
            src: dataframe 或者"list和路径的任意嵌套"
            disable: 是否禁用进度条
            chunksize: 批次处理的大小
            processes: 多进程数量
            postfix: 进度条后缀
            graph_only: 是否返回graph之外的东西
            mp: multiprocessing库的来源，有'dgl'，'torch'和'python'三种，如果是'auto'，那么会按照上述顺序try
            **kwargs:

        Returns:
            df: df which drop json cols
            graphs: dict of all2graph.Graph or str if temp_file == True
            global_index_mappers:
            local_index_mapperss
        """
        if mp == 'auto':
            try:
                from dgl.multiprocessing import Pool
            except ImportError:
                try:
                    from torch.multiprocessing import Pool
                except ImportError:
                    from multiprocessing import Pool
        elif mp == 'dgl':
            from dgl.multiprocessing import Pool
        elif mp == 'torch':
            from dgl.multiprocessing import Pool
        elif mp == 'python':
            from multiprocessing import Pool
        else:
            raise ValueError('mp should be one of ("auto", "dgl", "torch", "python")')

        def foo():
            if graph_only:
                return item[1]
            else:
                return item

        data = dataframe_chunk_iter(src, chunksize=chunksize, **kwargs)
        if processes == 0:
            for item in tqdm(map(self.parse, data), disable=disable, postfix=postfix):
                yield foo()
        else:
            with Pool(processes) as pool:
                for item in tqdm(pool.imap_unordered(self.parse, data), disable=disable, postfix=postfix):
                    yield foo()
