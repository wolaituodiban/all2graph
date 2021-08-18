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

    def nodes_to_df(self) -> pd.DataFrame:
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

    def to_df(self) -> (pd.DataFrame, pd.DataFrame):
        """
        分别返回节点dataframe和边dataframe
        """
        node_df = self.nodes_to_df()
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
        return node_df, edge_df


