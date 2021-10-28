import time
from typing import Tuple, List, Dict, Union

import pandas as pd

from .data import DataParser
from .graph import RawGraphParser
from ..graph import Graph


class ParserWrapper:
    def __init__(self,
                 data_parser: Union[DataParser, Dict[str, DataParser]], raw_graph_parser: RawGraphParser,
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
    ) -> Tuple[pd.DataFrame, Dict[str, Union[Graph, str]], Dict[str, dict], Dict[str, List[dict]]]:
        graphs = {}
        global_index_mappers = {}
        local_index_mapperss = {}
        for name, parser in self.data_parsers.items():
            raw_graph, global_index_mapper, local_index_mappers = parser.parse(df, progress_bar=not disable)
            graph = self.raw_graph_parser.parse(raw_graph)
            if self.temp_file:
                filename = str(time.time())
                filename = '+'.join([filename, str(id(filename))])
                graph.save(filename)
                graph = filename
            graphs[name] = graph
            global_index_mappers[name] = global_index_mapper
            local_index_mapperss[name] = local_index_mappers
        return df.drop(columns=self.json_cols), graphs, global_index_mappers, local_index_mapperss
