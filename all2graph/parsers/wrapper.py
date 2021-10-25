import time
from typing import Tuple, List

import pandas as pd

from .data import DataParser
from .graph import RawGraphParser
from ..graph import Graph


class ParserWrapper:
    def __init__(self, data_parser: DataParser, raw_graph_parser: RawGraphParser, temp_file=False):
        self.data_parser = data_parser
        self.raw_graph_parser = raw_graph_parser
        self.temp_file = temp_file

    def parser(self, df: pd.DataFrame, disable=True) -> Tuple[pd.DataFrame, Graph, dict, List[dict]]:
        raw_graph, global_index_mapper, local_index_mappers = self.data_parser.parse(df, progress_bar=not disable)
        graph = self.raw_graph_parser.parse(raw_graph)
        if self.temp_file:
            filename = str(time.time())
            filename = '+'.join([filename, str(id(filename))])
            graph.save(filename)
            graph = filename
        return df.drop(columns=self.data_parser.json_col), graph, global_index_mapper, local_index_mappers
