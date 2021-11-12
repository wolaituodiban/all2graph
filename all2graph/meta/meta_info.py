from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd

from .meta_value import MetaNumber, MetaString
from ..graph import RawGraph
from ..meta_struct import MetaStruct
from ..globals import EPSILON, COMPONENT_ID
from ..preserves import NUMBER
from ..preserves import NULL, TRUE, FALSE, KEY, VALUE
from ..stats import ECDF
from ..utils import tqdm


class MetaInfo(MetaStruct):
    def __init__(
            self,
            meta_string: MetaString,
            meta_numbers: Dict[str, MetaNumber],
            meta_name: MetaString,
            edge_type: Set[Tuple[str, str]],
            **kwargs
    ):
        super().__init__(**kwargs)
        self.meta_string = meta_string
        self.meta_numbers = meta_numbers
        self.meta_name = meta_name
        self.edge_type = edge_type

    @property
    def num_strings(self):
        return len(self.meta_string)

    @property
    def num_numbers(self):
        return len(self.meta_numbers)

    @property
    def num_keys(self):
        return len(self.meta_name)

    @property
    def num_etypes(self):
        return len(self.edge_type)

    def __eq__(self, other, debug=False):
        if not super().__eq__(other):
            if debug:
                print('super not equal')
            return False
        if self.meta_numbers != other.meta_numbers:
            if debug:
                print('meta_numebrs not equal')
            return False
        if self.meta_name != other.meta_name:
            if debug:
                print('meta_name not equal')
            return False
        if self.meta_string != other.meta_string:
            if debug:
                print('meta_string not equal')
            return False
        if self.edge_type != other.edge_type:
            if debug:
                print('edge_type not equal')
            return False
        return True

    def to_json(self) -> dict:
        output = super().to_json()
        output['meta_string'] = self.meta_string.to_json()
        output['meta_numbers'] = {k: v.to_json() for k, v in self.meta_numbers.items()}
        output['meta_name'] = self.meta_name.to_json()
        output['edge_type'] = self.edge_type
        return output

    @classmethod
    def from_json(cls, obj: dict):
        obj = dict(obj)
        obj['meta_string'] = MetaString.from_json(obj['meta_string'])
        obj['meta_numbers'] = {k: MetaNumber.from_json(v) for k, v in obj['meta_numbers'].items()}
        obj['meta_name'] = MetaString.from_json(obj['meta_name'])
        return super().from_json(obj)

    @classmethod
    def from_data(cls, graph: RawGraph, index_nodes=None, disable=True, num_bins=None):
        node_df = pd.DataFrame(
            {
                COMPONENT_ID: graph.component_id,
                KEY: graph.key,
                VALUE: graph.value,
            }
        )
        node_df[COMPONENT_ID] = node_df[COMPONENT_ID].abs()
        num_samples = node_df[COMPONENT_ID].unique().shape[0]

        edge_type = set(graph.edge_key)

        # # # # # 生成meta_name # # # # #
        meta_name = MetaString.from_data(
            num_samples=num_samples, sample_ids=node_df[COMPONENT_ID], values=node_df[KEY],
            disable=disable, postfix='constructing meta name', num_bins=num_bins
        )

        # # # # # 生成meta_numbers # # # # #
        if index_nodes is not None:
            node_df = node_df.drop(index_nodes)
        node_df[NUMBER] = pd.to_numeric(node_df[VALUE], errors='coerce')
        number_df = node_df[np.isfinite(node_df[NUMBER])]

        number_groups = number_df.groupby(KEY, sort=False)
        number_groups = tqdm(number_groups, disable=disable, postfix='constructing meta numbers')
        meta_numbers = {}
        for name, number_df in number_groups:
            meta_numbers[name] = MetaNumber.from_data(
                num_samples=num_samples, sample_ids=number_df[COMPONENT_ID], values=number_df[NUMBER], num_bins=num_bins
            )

        # # # # # 生成meta_string # # # # #
        node_df = node_df[pd.isna(node_df[NUMBER]) & node_df[VALUE].apply(lambda x: not isinstance(x, (dict, list)))]

        def bool_to_str(x):
            if isinstance(x, bool):
                return TRUE if x else FALSE
            else:
                return x

        node_df[VALUE] = node_df[VALUE].apply(bool_to_str)
        node_df[VALUE] = node_df[VALUE].fillna(NULL)

        meta_string = MetaString.from_data(
            num_samples=num_samples, sample_ids=node_df[COMPONENT_ID], values=node_df[VALUE],
            disable=disable, num_bins=num_bins
        )

        return super().from_data(meta_string=meta_string, meta_numbers=meta_numbers, meta_name=meta_name,
                                 edge_type=edge_type)

    @classmethod
    def reduce(cls, structs, weights=None, disable=True, num_bins=None, processes=None):
        if weights is None:
            weights = np.full(len(structs), 1 / len(structs))
        else:
            weights = np.array(weights) / sum(weights)

        # # # # # 合并meta_numbers # # # # #
        meta_numbers = {}
        meta_num_w = {}
        for w, struct in zip(weights, structs):
            for k, v in struct.meta_numbers.items():
                if k not in meta_numbers:
                    meta_numbers[k] = [v]
                    meta_num_w[k] = [w]
                else:
                    meta_numbers[k].append(v)
                    meta_num_w[k].append(w)

        progress = tqdm(meta_numbers.items(), disable=disable, postfix='reducing meta numbers')
        meta_numbers = {
            k: MetaNumber.reduce(v, weights=meta_num_w[k], num_bins=num_bins) for k, v in progress
        }

        # 分布补0
        progress = tqdm(meta_numbers, disable=disable, postfix='reducing meta numbers phase 2')
        for k in progress:
            w_sum = sum(meta_num_w[k])
            if w_sum < 1 - EPSILON:
                meta_numbers[k].count_ecdf = ECDF.reduce(
                    [meta_numbers[k].count_ecdf, ECDF([0], [1], initialized=True)],
                    weights=[w_sum, 1-w_sum], num_bins=num_bins
                )
        # # # # # 合并meta_string # # # # #
        meta_string = MetaString.reduce(
            [struct.meta_string for struct in structs], weights=weights, disable=disable, processes=processes,
            num_bins=num_bins
        )
        # # # # # 合并meta_name # # # # #
        meta_name = MetaString.reduce(
            [struct.meta_name for struct in structs], weights=weights, disable=disable,
            postfix='reducing meta name', processes=processes, num_bins=num_bins
        )
        # # # # # 合并edge_type # # # # #
        edge_type = set()
        for struct in structs:
            edge_type = edge_type.union(struct.edge_type)

        return super().reduce(
            structs, weights=weights, meta_numbers=meta_numbers, meta_string=meta_string, meta_name=meta_name,
            edge_type=edge_type,
        )

    def extra_repr(self) -> str:
        return 'num_strings={}, num_numbers={}, num_keys={}, num_etypes={}'.format(
            self.meta_string, self.num_numbers, self.num_keys, self.num_etypes
        )
