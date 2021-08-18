from typing import Dict

import numpy as np
import pandas as pd
from toad.utils.progress import Progress

from ..graph import Graph
from ..meta_struct import MetaStruct
from ..meta_graph import MetaNumber, MetaString
from ..macro import NULL, TRUE, FALSE
from ..stats import ECDF


class GraphDecoder(MetaStruct):
    def __init__(self, meta_string: MetaString, meta_numbers: Dict[str, MetaNumber], **kwargs):
        super().__init__(**kwargs)
        self.meta_string = meta_string
        self.meta_numbers = meta_numbers

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
    def from_data(cls, graph: Graph, **kwargs):
        node_df = graph.nodes_to_df()
        num_samples = node_df.component_id.unique().shape[0]

        node_df['number'] = pd.to_numeric(node_df.value, errors='coerce')
        node_df.loc[pd.notna(node_df.number), 'value'] = NULL
        node_df.loc[node_df.value.apply(lambda x: isinstance(x, (dict, list))), 'value'] = NULL

        def bool_to_str(x):
            if isinstance(x, bool):
                return TRUE if x else FALSE
            else:
                return x
        node_df['value'] = node_df.value.apply(bool_to_str)
        node_df['value'] = node_df['value'].fillna(NULL)

        meta_string = MetaString.from_data(
            num_samples=num_samples, sample_ids=node_df.component_id, values=node_df.value, **kwargs
        )

        meta_numbers = {}
        for name, df in node_df.dropna(subset=['number']).groupby('name'):
            meta_numbers[name] = MetaNumber.from_data(
                num_samples=num_samples, sample_ids=df.component_id, values=df.number, **kwargs
            )
        return super().from_data(meta_string=meta_string, meta_numbers=meta_numbers, **kwargs)

    @classmethod
    def reduce(cls, structs, weights=None, **kwargs):
        if weights is None:
            weights = np.full(len(structs), 1 / len(structs))
        else:
            weights = np.array(weights) / sum(weights)

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

        meta_numbers = {
            k: MetaNumber.reduce(v, weights=meta_num_w[k], **kwargs) for k, v in Progress(meta_numbers.items())
        }

        for k in Progress(meta_numbers):
            w_sum = sum(meta_num_w[k])
            if w_sum < 1:
                meta_numbers[k].freq = ECDF.reduce(
                    [meta_numbers[k].freq, ECDF([0], [1], initialized=True)],
                    weights=[w_sum, 1-w_sum], **kwargs
                )

        meta_string = MetaString.reduce([struct.meta_string for struct in structs], weights=weights, **kwargs)
        return super().reduce(structs, weights=weights, meta_numbers=meta_numbers, meta_string=meta_string, **kwargs)
