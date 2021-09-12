from typing import Dict

import numpy as np
import pandas as pd

from .meta_node import MetaNumber, MetaString
from ..graph import Graph
from ..meta_struct import MetaStruct
from ..globals import NULL, TRUE, FALSE, EPSILON, COMPONENT_ID, KEY, VALUE, NUMBER
from ..stats import ECDF
from ..utils import progress_wrapper


class MetaGraph(MetaStruct):
    def __init__(
            self,
            meta_string: MetaString,
            meta_numbers: Dict[str, MetaNumber],
            meta_name: MetaString,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.meta_string = meta_string
        self.meta_numbers = meta_numbers
        self.meta_name = meta_name

    def __eq__(self, other):
        return super().__eq__(other)\
               and self.meta_string == other.meta_string\
               and self.meta_numbers == other.meta_numbers\
               and self.meta_name == other.meta_name

    def to_json(self) -> dict:
        output = super().to_json()
        output['meta_string'] = self.meta_string.to_json()
        output['meta_numbers'] = {k: v.to_json() for k, v in self.meta_numbers.items()}
        output['meta_name'] = self.meta_name.to_json()
        return output

    @classmethod
    def from_json(cls, obj: dict):
        obj = dict(obj)
        obj['meta_string'] = MetaString.from_json(obj['meta_string'])
        obj['meta_numbers'] = {k: MetaNumber.from_json(v) for k, v in obj['meta_numbers'].items()}
        obj['meta_name'] = MetaString.from_json(obj['meta_name'])
        return super().from_json(obj)

    @classmethod
    def from_data(cls, graph: Graph, index_nodes=None, progress_bar=False, **kwargs):
        node_df = pd.DataFrame(
            {
                COMPONENT_ID: graph.component_id,
                KEY: graph.key,
                VALUE: graph.value,
            }
        )
        node_df[COMPONENT_ID] = node_df[COMPONENT_ID].abs()
        num_samples = node_df[COMPONENT_ID].unique().shape[0]

        # # # # # 生成meta_name # # # # #
        meta_name = MetaString.from_data(
            num_samples=num_samples, sample_ids=node_df[COMPONENT_ID], values=node_df[KEY],
            progress_bar=progress_bar, postfix='constructing meta name', **kwargs
        )

        # # # # # 生成meta_numbers # # # # #
        if index_nodes is not None:
            node_df = node_df.drop(index_nodes)
        node_df[NUMBER] = pd.to_numeric(node_df[VALUE], errors='coerce')
        number_df = node_df[np.isfinite(node_df[NUMBER])]

        number_groups = number_df.groupby(KEY, sort=False)
        number_groups = progress_wrapper(number_groups, disable=not progress_bar, postfix='constructing meta numbers')
        meta_numbers = {}
        for name, number_df in number_groups:
            meta_numbers[name] = MetaNumber.from_data(
                num_samples=num_samples, sample_ids=number_df[COMPONENT_ID], values=number_df[NUMBER], **kwargs
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
            progress_bar=progress_bar, **kwargs
        )

        return super().from_data(meta_string=meta_string, meta_numbers=meta_numbers, meta_name=meta_name, **kwargs)

    @classmethod
    def reduce(cls, structs, weights=None, progress_bar=False, **kwargs):
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

        progress = progress_wrapper(meta_numbers.items(), disable=not progress_bar, postfix='reducing meta numbers')
        meta_numbers = {
            k: MetaNumber.reduce(v, weights=meta_num_w[k], **kwargs) for k, v in progress
        }

        # 分布补0
        progress = progress_wrapper(meta_numbers, disable=not progress_bar, postfix='reducing meta numbers phase 2')
        for k in progress:
            w_sum = sum(meta_num_w[k])
            if w_sum < 1 - EPSILON:
                meta_numbers[k].count_ecdf = ECDF.reduce(
                    [meta_numbers[k].count_ecdf, ECDF([0], [1], initialized=True)],
                    weights=[w_sum, 1-w_sum], **kwargs
                )
        # # # # # 合并meta_string # # # # #
        meta_string = MetaString.reduce(
            [struct.meta_string for struct in structs], weights=weights, progress_bar=progress_bar, **kwargs
        )
        # # # # # 合并meta_name # # # # #
        meta_name = MetaString.reduce(
            [struct.meta_name for struct in structs], weights=weights, progress_bar=progress_bar,
            postfix='reducing meta name', **kwargs
        )
        return super().reduce(
            structs, weights=weights, meta_numbers=meta_numbers, meta_string=meta_string, meta_name=meta_name, **kwargs
        )
