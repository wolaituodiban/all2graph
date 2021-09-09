from typing import Dict, List, Union, Tuple

import numpy as np

from ..globals import COMPONENT_ID, KEY, VALUE, SRC, DST
from ..meta_struct import MetaStruct


class Graph(MetaStruct):
    def __init__(self, component_id=None, key=None, values=None, src=None, dst=None):
        super().__init__(initialized=True)
        self.component_id: List[int] = list(component_id or [])
        self.key: List[str] = list(key or [])
        self.values: List[Union[Dict, List, str, int, float, None]] = list(values or [])
        self.src: List[int] = list(src or [])
        self.dst: List[int] = list(dst or [])

        assert len(self.component_id) == len(self.key) == len(self.values)
        assert len(self.src) == len(self.dst)

    def __eq__(self, other):
        return super().__eq__(other) \
               and self.component_id == other.component_id \
               and self.key == other.key \
               and self.values == other.values \
               and self.src == other.src \
               and self.dst == other.dst

    def to_json(self) -> dict:
        output = super().to_json()
        output[COMPONENT_ID] = self.component_id
        output[KEY] = self.key
        output[VALUE] = self.values
        output[SRC] = self.src
        output[DST] = self.dst
        return output

    @classmethod
    def from_json(cls, obj: dict):
        return super().from_json(obj)

    @property
    def num_nodes(self):
        return len(self.key)

    @property
    def num_edges(self):
        return len(self.src)

    @property
    def num_components(self):
        return np.unique(self.component_id).shape[0]

    def insert_edges(self, preds: List[int], succs: List[int]):
        self.src += preds
        self.dst += succs

    def insert_node(
            self,
            patch_id: int,
            name: str,
            value: Union[Dict, List, str, int, float, None],
            self_loop: bool
    ) -> int:
        node_id = len(self.key)
        self.component_id.append(patch_id)
        self.key.append(name)
        self.values.append(value)
        if self_loop:
            self.src.append(node_id)
            self.dst.append(node_id)
        return node_id

    def meta_node_info(self) -> Tuple[
        List[int], Dict[Tuple[int, str], int], List[int], List[str]
    ]:
        meta_node_ids: List[int] = []
        meta_node_id_mapper: Dict[Tuple[int, str], int] = {}
        meta_node_component_ids: List[int] = []
        meta_node_names: List[str] = []
        for i, name in zip(self.component_id, self.key):
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
        for pred, succ in zip(self.src, self.dst):
            component_id, pred_name, succ_name = self.component_id[pred], self.key[pred], self.key[succ]
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
        component_ids = [struct.component_id for struct in structs]
        names = [struct.key for struct in structs]
        values = [struct.values for struct in structs]
        preds = [struct.src for struct in structs]
        succs = [struct.dst for struct in structs]
        return super().reduce(
            structs, component_ids=component_ids, names=names, values=values, preds=preds, succs=succs, **kwargs
        )
