from abc import abstractmethod
from typing import Tuple, List
import numpy as np
import pandas as pd
from ..graph import RawGraph
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
            for k in target_cols if k in df
        }

    @abstractmethod
    def parse(self, data, disable: bool = True) -> (RawGraph, dict, List[dict]):
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


class DataAugmenter(DataParser):
    def __init__(self, parsers: List[DataParser], weights: List[float] = None):
        super(DataParser, self).__init__(initialized=True)
        self.parsers = parsers
        if weights is None:
            self.weights = np.ones(len(self.parsers)) / len(self.parsers)
        else:
            self.weights = np.array(weights) / np.sum(weights)

    def parse(self, data, disable: bool = True, **kwargs) -> Tuple[RawGraph, dict, List[dict]]:
        parser = np.random.choice(self.parsers, p=self.weights)
        return parser.parse(data, disable=disable, **kwargs)
