import os
from multiprocessing import Pool
from typing import Dict, Union, Iterable

import numpy as np
import pandas as pd

from .number_info import NumberInfo
from .token_info import TokenInfo
from ...meta_struct import MetaStruct
from ...globals import EPSILON
from ...stats import ECDF
from ...utils import MpMapFuncWrapper, tqdm


class StringInfo(MetaStruct):
    """类别节点"""
    def __init__(self, number_info: Dict[str, NumberInfo], token_info: Dict[str, TokenInfo], **kwargs):
        super().__init__(**kwargs)
        self.number_info = number_info
        self.token_info = token_info

    def __eq__(self, other, debug=False):
        if not super().__eq__(other):
            if debug:
                print('super not equal')
            return False
        if self.number_info != other.number_info:
            if debug:
                print('number_info not equal')
            return False
        if self.token_info != other.token_info:
            if debug:
                print('token_info not equal')
            return False
        return True

    @property
    def doc_freq(self) -> Dict[str, float]:
        return {token: info.doc_freq for token, info in self.token_info.items()}

    @property
    def tf_idf(self) -> Dict[str, ECDF]:
        return {token: info.tf_idf for token, info in self.token_info.items()}

    def to_json(self) -> dict:
        output = super().to_json()
        output['number_info'] = {k: v.to_json() for k, v in self.number_info.items()}
        output['token_info'] = {k: v.to_json() for k, v in self.token_info.items()}
        return output

    @classmethod
    def from_json(cls, obj):
        obj = dict(obj)
        obj['number_info'] = {k: ECDF.from_json(v) for k, v in obj['number_info'].items()}
        obj['token_info'] = {k: ECDF.from_json(v) for k, v in obj['token_info'].items()}
        return super().from_json(obj)

    @classmethod
    def from_data(cls, sample_ids, keys, values, num_bins=None, disable=True, postfix='constructing info string'):


    @classmethod
    def reduce(cls, structs, weights=None, num_bins=None, processes=0, chunksize=None,
               disable=True, postfix='reducing info string'):
        raise NotImplemented

    def extra_repr(self) -> str:
        return 'number_info={}\ntoken_info={}'.format(self.number_info, self.token_info)
