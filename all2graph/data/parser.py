from typing import Tuple, List
import pandas as pd
from ..graph import Graph
from ..meta_struct import MetaStruct


class DataParser(MetaStruct):
    def __init__(self, json_col, time_col, time_format, **kwargs):
        super().__init__(initialized=True, **kwargs)
        self.json_col = json_col
        self.time_col = time_col
        self.time_format = time_format

    @staticmethod
    def gen_targets(df: pd.DataFrame, target_cols):
        import torch
        return {
            k: torch.tensor(pd.to_numeric(df[k], errors='coerce').values, dtype=torch.float32)
            for k in target_cols
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
