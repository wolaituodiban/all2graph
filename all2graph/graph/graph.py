from typing import Dict, List, Union

import numpy as np
import pandas as pd

from ..meta_struct import MetaStruct


class Graph(MetaStruct):
    def __init__(self, components_ids=None, names=None, values=None, preds=None, succs=None):
        super().__init__(initialized=True)
        self.component_ids: List[int] = components_ids or []
        self.names: List[str] = names or []
        self.values: List[Union[Dict, List, str, int, float, None]] = values or []
        self.preds: List[int] = preds or []
        self.succs: List[int] = succs or []

    def __eq__(self, other):
        return super().__eq__(other)\
               and self.component_ids == other.component_ids\
               and self.names == other.names\
               and self.values == other.values\
               and self.preds == other.preds\
               and self.succs == other.succs

    def to_json(self) -> dict:
        output = super().to_json()
        output['component_ids'] = self.component_ids
        output['names'] = self.names
        output['values'] = self.values
        output['preds'] = self.preds
        output['succs'] = self.succs
        return output

    @classmethod
    def from_json(cls, obj: dict):
        return super().from_json(obj)

    @property
    def num_nodes(self):
        assert len(self.names) == len(self.values)
        return len(self.names)

    @property
    def num_edges(self):
        assert len(self.preds) == len(self.succs)
        return len(self.preds)

    @property
    def num_components(self):
        return np.unique(self.component_ids).shape[0]

    def insert_edges(self, preds: List[int], succs: List[int]):
        self.preds += preds
        self.succs += succs

    def insert_node(
            self,
            patch_id: int,
            name: str,
            value: Union[Dict, List, str, int, float, None],
    ) -> int:
        node_id = len(self.names)
        self.component_ids.append(patch_id)
        self.names.append(name)
        self.values.append(value)
        return node_id

    def node_df(self) -> pd.DataFrame:
        """
        将节点信息以dataframe的形式返回
        """
        return pd.DataFrame(
            {
                'component_id': self.component_ids,
                'name': self.names,
                'value': self.values,
            }
        )

    def edge_df(self, node_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        分别返回节点dataframe和边dataframe
        """
        node_df = node_df or self.node_df()
        edge_df = pd.DataFrame(
            {
                'component_id': node_df.component_id.iloc[self.preds],
                'pred': self.preds,
                'pred_name': node_df.name.iloc[self.preds],
                'succ': self.succs,
                'succ_name': node_df.name.iloc[self.succs]
            }
        )
        edge_df = edge_df.merge(node_df, left_on='succ', right_index=True, how='left')
        return edge_df

    def meta_df(self, node_df: pd.DataFrame = None, edge_df: pd.DataFrame = None) -> pd.DataFrame:
        edge_df = edge_df or self.edge_df(node_df)
        meta_df = []
        for component_id, component_df in edge_df.groupby('component_id'):
            component_df = component_df[['pred_name', 'succ_name']].drop_duplicates()
            meta_df.append(component_df)
        meta_df = pd.concat(meta_df)
        return meta_df

    @classmethod
    def from_data(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def reduce(cls, structs, **kwargs):
        component_ids = [struct.component_ids for struct in structs]
        names = [struct.names for struct in structs]
        values = [struct.values for struct in structs]
        preds = [struct.preds for struct in structs]
        succs = [struct.succs for struct in structs]
        return super().reduce(
            structs, component_ids=component_ids, names=names, values=values, preds=preds, succs=succs, **kwargs
        )
