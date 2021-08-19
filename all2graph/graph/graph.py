from typing import Dict, List, Union

import numpy as np
import pandas as pd


class Graph:
    def __init__(self):
        self.component_ids: List[int] = []
        self.names: List[str] = []
        self.values: List[Union[Dict, List, str, int, float, None]] = []
        self.preds: List[int] = []
        self.succs: List[int] = []

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
