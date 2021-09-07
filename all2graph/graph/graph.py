from typing import Dict, List, Union, Tuple

import numpy as np

from ..globals import COMPONENT_ID, NAME, VALUE
from ..meta_struct import MetaStruct


class Graph(MetaStruct):
    def __init__(self, component_ids=None, names=None, values=None, preds=None, succs=None):
        super().__init__(initialized=True)
        self.component_ids: List[int] = list(component_ids or [])
        self.names: List[str] = list(names or [])
        self.values: List[Union[Dict, List, str, int, float, None]] = list(values or [])
        self.preds: List[int] = list(preds or [])
        self.succs: List[int] = list(succs or [])

        assert len(self.component_ids) == len(self.names) == len(self.values)
        assert len(self.preds) == len(self.succs)

    def __eq__(self, other):
        return super().__eq__(other)\
               and self.component_ids == other.component_ids\
               and self.names == other.names\
               and self.values == other.values\
               and self.preds == other.preds\
               and self.succs == other.succs

    def to_json(self) -> dict:
        output = super().to_json()
        output[COMPONENT_ID] = self.component_ids
        output[NAME] = self.names
        output[VALUE] = self.values
        output['preds'] = self.preds
        output['succs'] = self.succs
        return output

    @classmethod
    def from_json(cls, obj: dict):
        return super().from_json(obj)

    @property
    def num_nodes(self):
        return len(self.names)

    @property
    def num_edges(self):
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

    def meta_node_info(self) -> Tuple[
        List[int], Dict[Tuple[int, str], int], List[int], List[str]
    ]:
        meta_node_ids: List[int] = []
        meta_node_id_mapper: Dict[Tuple[int, str], int] = {}
        meta_node_component_ids: List[int] = []
        meta_node_names: List[str] = []
        for i, name in zip(self.component_ids, self.names):
            if (i, name) not in meta_node_id_mapper:
                meta_node_id_mapper[(i, name)] = len(meta_node_id_mapper)
                meta_node_component_ids.append(i)
                meta_node_names.append(name)
            meta_node_ids.append(meta_node_id_mapper[(i, name)])
        return meta_node_ids, meta_node_id_mapper, meta_node_component_ids, meta_node_names

    def meta_edge_info(self, meta_node_id_mapper: Dict[Tuple[int, str], int]) -> Tuple[
        List[int], List[int], List[int]
    ]:
        meta_edge_ids: List[int] = []
        meta_edge_id_mapper: Dict[Tuple[int, str, str], int] = {}
        pred_meta_node_ids: List[int] = []
        succ_meta_node_ids: List[int] = []
        for pred, succ in zip(self.preds, self.succs):
            component_id, pred_name, succ_name = self.component_ids[pred], self.names[pred], self.names[succ]
            if (component_id, pred_name, succ_name) not in meta_edge_id_mapper:
                meta_edge_id_mapper[(component_id, pred_name, succ_name)] = len(meta_edge_id_mapper)
                pred_meta_node_ids.append(meta_node_id_mapper[(component_id, pred_name)])
                succ_meta_node_ids.append(meta_node_id_mapper[(component_id, succ_name)])
            meta_edge_ids.append(meta_edge_id_mapper[(component_id, pred_name, succ_name)])
        return meta_edge_ids, pred_meta_node_ids, succ_meta_node_ids

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
