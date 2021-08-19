from typing import Dict

import numpy as np
import pandas as pd
import jieba
from toad.utils.progress import Progress

from ..graph import Graph
from ..meta_struct import MetaStruct
from ..meta_graph import MetaNumber, MetaString
from ..macro import NULL, TRUE, FALSE
from ..stats import ECDF


class GraphDecoder(MetaStruct):
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
               and self.meta_numbers == other.meta_numbers

    def to_json(self) -> dict:
        output = super().to_json()
        output['meta_string'] = self.meta_string.to_json()
        output['meta_numbers'] = {k: v.to_json() for k, v in self.meta_numbers.items()}
        return output

    @classmethod
    def from_json(cls, obj: dict):
        obj = dict(obj)
        obj['meta_string'] = MetaString.from_json(obj['meta_string'])
        obj['meta_numbers'] = {k: MetaNumber.from_json(v) for k, v in obj['meta_numbers'].items()}
        return super().from_json(**obj)

    @classmethod
    def from_data(cls, graph: Graph, index_ids=None, progress_bar=False, **kwargs):
        node_df = graph.node_df()
        num_samples = node_df.component_id.unique().shape[0]

        if index_ids is not None:
            node_df = node_df.drop(index_ids)

        # # # # # 生成meta_numbers # # # # #
        node_df['number'] = pd.to_numeric(node_df.value, errors='coerce')
        meta_numbers = {}
        progress = node_df.dropna(subset=['number']).groupby('name', sort=False)
        if progress_bar:
            progress = Progress(progress)
            progress.prefix = 'constructing meta numbers'
        for name, df in progress:
            meta_numbers[name] = MetaNumber.from_data(
                num_samples=num_samples, sample_ids=df.component_id, values=df.number, **kwargs
            )

        # # # # # 生成meta_string # # # # #
        node_df = node_df[pd.isna(node_df.number) & node_df.value.apply(lambda x: not isinstance(x, (dict, list)))]

        def bool_to_str(x):
            if isinstance(x, bool):
                return TRUE if x else FALSE
            else:
                return x

        node_df['value'] = node_df.value.apply(bool_to_str)
        node_df['value'] = node_df.value.fillna(NULL)

        meta_string = MetaString.from_data(
            num_samples=num_samples, sample_ids=node_df.component_id, values=node_df.value, progress_bar=progress_bar,
            **kwargs
        )

        # # # # # 生成meta_name # # # # #
        name_split = {
            name: jieba.lcut(name) for name in node_df['name'].unique()
        }
        token_sample_ids = []
        tokens = []

        progress = node_df.groupby('component_id', sort=False)
        if progress_bar:
            progress = Progress(progress)
            progress.prefix = 'constructing meta name'

        for component_id, component in progress:
            for name in component['name'].unique():
                for token in name_split[name]:
                    token_sample_ids.append(component_id)
                    tokens.append(token)

        meta_name = MetaString.from_data(
            num_samples=num_samples, sample_ids=token_sample_ids, values=tokens, progress_bar=progress_bar,
            prefix='constructing meta name phase 2', **kwargs
        )

        return super().from_data(meta_string=meta_string, meta_numbers=meta_numbers, meta_name=meta_name, **kwargs)

    @classmethod
    def reduce(cls, structs, weights=None, progress_bar=True, **kwargs):
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

        progress = meta_numbers.items()
        if progress_bar:
            progress = Progress(progress)
            progress.prefix = 'reducing meta numbers'
        meta_numbers = {
            k: MetaNumber.reduce(v, weights=meta_num_w[k], **kwargs) for k, v in progress
        }

        # 分布补0
        progress = meta_numbers
        if progress_bar:
            progress = Progress(progress)
            progress.prefix = 'reducing meta numbers phase 2'
        for k in progress:
            w_sum = sum(meta_num_w[k])
            if w_sum < 1:
                meta_numbers[k].freq = ECDF.reduce(
                    [meta_numbers[k].freq, ECDF([0], [1], initialized=True)],
                    weights=[w_sum, 1-w_sum], **kwargs
                )
        # # # # # 合并meta_string # # # # #
        meta_string = MetaString.reduce(
            [struct.meta_string for struct in structs], weights=weights, progress_bar=progress_bar, **kwargs
        )
        # # # # # 合并meta_name # # # # #
        meta_name = MetaString.reduce(
            [struct.meta_name for struct in structs], weights=weights, progress_bar=progress_bar,
            prefix='reducing meta name', **kwargs
        )
        return super().reduce(
            structs, weights=weights, meta_numbers=meta_numbers, meta_string=meta_string, meta_name=meta_name, **kwargs
        )
