from typing import Dict, List, Union, Tuple

import numpy as np
import pandas as pd

from ..meta_struct import MetaStruct


class Graph(MetaStruct):
    def __init__(self, component_ids=None, names=None, values=None, preds=None, succs=None):
        super().__init__(initialized=True)
        self.component_ids: List[int] = component_ids or []
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
            self_loop: bool
    ) -> int:
        node_id = len(self.names)
        self.component_ids.append(patch_id)
        self.names.append(name)
        self.values.append(value)
        if self_loop:
            self.preds.append(node_id)
            self.succs.append(node_id)
        return node_id

    def meta_nodes(self) -> Tuple[np.ndarray, np.ndarray]:
        """

        :return: np.ndarray, np.ndarray
            元点点分片编号(component_id)，元点的名字(name)
        """
        df = pd.DataFrame([self.component_ids, self.names]).drop_duplicates()
        return df[0].values, df[1].values

    def meta_edges(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        edge_component_ids = [self.component_ids[i] for i in self.preds]
        edge_pred_names = [self.component_ids[i] for i in self.preds]
        edge_succ_names = [self.component_ids[i] for i in self.succs]
        df = pd.DataFrame([edge_component_ids, edge_pred_names, edge_succ_names]).drop_duplicates()
        return df[0].values, df[1].values, df[2].values

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
        if node_df is None:
            node_df = self.node_df()
        edge_df = pd.DataFrame(
            {
                'component_id': node_df.component_id.iloc[self.preds].values,
                'pred': self.preds,
                'pred_name': node_df.name.iloc[self.preds].values,
                'succ': self.succs,
                'succ_name': node_df.name.iloc[self.succs].values
            }
        )
        return edge_df

    def meta_node_df(self, node_df: pd.DataFrame = None) -> pd.DataFrame:
        if node_df is None:
            node_df = self.node_df()
        df = node_df[['component_id', 'name']].drop_duplicates()
        df['meta_node_id'] = np.arange(0, df.shape[0], 1)
        return df

    def meta_edge_df(self, node_df: pd.DataFrame = None, edge_df: pd.DataFrame = None) -> pd.DataFrame:
        if edge_df is None:
            edge_df = self.edge_df(node_df)
        df = edge_df[['component_id', 'pred_name', 'succ_name']].drop_duplicates()
        df['meta_edge_id'] = np.arange(0, df.shape[0], 1)
        return df

    @classmethod
    def from_data(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def reduce(cls, structs, **kwargs):
        raise NotImplementedError

    @classmethod
    def merge(cls, structs, **kwargs):
        component_ids = [struct.component_ids for struct in structs]
        names = [struct.names for struct in structs]
        values = [struct.values for struct in structs]
        preds = [struct.preds for struct in structs]
        succs = [struct.succs for struct in structs]
        return super().reduce(
            structs, component_ids=component_ids, names=names, values=values, preds=preds, succs=succs, **kwargs
        )
