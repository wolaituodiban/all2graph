from typing import Tuple, List
import pandas as pd
from ..graph import Graph
from ..globals import TARGET
from ..meta_struct import MetaStruct


class DataParser(MetaStruct):
    def __init__(self, json_col, time_col, time_format, target_cols, **kwargs):
        super().__init__(initialized=True, **kwargs)
        self.json_col = json_col
        self.time_col = time_col
        self.time_format = time_format
        self.target_cols = list(target_cols or [])

    @staticmethod
    def add_targets(graph: Graph, component_id, readout_id, targets):
        for target in targets:
            target_id = graph.insert_node(component_id, target, value=None, self_loop=False, type=TARGET)
            graph.insert_edges([readout_id], [target_id])

    def gen_targets(self, df: pd.DataFrame):
        import torch
        return {
            k: torch.tensor(pd.to_numeric(df[k], errors='coerce').values, dtype=torch.float32)
            for k in self.target_cols
        }

    def parse(self, data, progress_bar: bool = False, **kwargs) -> Tuple[Graph, dict, List[dict]]:
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def to_json(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_json(cls, obj: dict):
        raise NotImplementedError

    @classmethod
    def from_data(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def reduce(cls, structs, weights=None, **kwargs):
        raise NotImplementedError
