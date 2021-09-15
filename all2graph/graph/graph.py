from typing import Dict, List, Union, Tuple

import jieba
import numpy as np

from ..globals import COMPONENT_ID, KEY, VALUE, SRC, DST, META, TYPE
from ..meta_struct import MetaStruct


class Graph(MetaStruct):
    def __init__(self, component_id=None, key=None, value=None, src=None, dst=None, type=None):
        super().__init__(initialized=True)
        self.component_id: List[int] = list(component_id or [])
        self.key: List[str] = list(key or [])
        self.value: List[Union[Dict, List, str, int, float, None]] = list(value or [])
        self.src: List[int] = list(src or [])
        self.dst: List[int] = list(dst or [])
        self.type: List[str] = list(type or [])

        assert len(self.component_id) == len(self.key) == len(self.value) == len(self.type)
        assert len(self.src) == len(self.dst)

    def __eq__(self, other):
        return super().__eq__(other) \
               and self.component_id == other.component_id \
               and self.key == other.key \
               and self.value == other.value \
               and self.src == other.src \
               and self.dst == other.dst \
               and self.type == other.type

    def to_json(self) -> dict:
        output = super().to_json()
        output[COMPONENT_ID] = self.component_id
        output[KEY] = self.key
        output[VALUE] = self.value
        output[SRC] = self.src
        output[DST] = self.dst
        output[TYPE] = self.type
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

    def insert_edges(self, srcs: List[int], dsts: List[int]):
        self.src += srcs
        self.dst += dsts

    def insert_node(
            self,
            component_id: int,
            name: str,
            value: Union[dict, list, str, int, float, bool, None],
            self_loop: bool,
            type=VALUE
    ) -> int:
        """

        :param component_id:
        :param name:
        :param value:
        :param self_loop:
        :param type:
        :return:
        """
        node_id = len(self.key)
        self.component_id.append(component_id)
        self.key.append(name)
        self.value.append(value)
        self.type.append(type)
        if self_loop:
            self.src.append(node_id)
            self.dst.append(node_id)
        return node_id

    def meta_node_info(self) -> Tuple[
        List[int], Dict[Tuple[int, str], int], List[int], List[str], List[str]
    ]:
        meta_node_ids: List[int] = []
        meta_node_id_mapper: Dict[Tuple[int, str], int] = {}
        meta_node_component_ids: List[int] = []
        meta_node_keys: List[str] = []
        for i, key in zip(self.component_id, self.key):
            if (i, key) not in meta_node_id_mapper:
                meta_node_id_mapper[(i, key)] = len(meta_node_id_mapper)
                meta_node_component_ids.append(i)
                meta_node_keys.append(key)
            meta_node_ids.append(meta_node_id_mapper[(i, key)])
        return meta_node_ids, meta_node_id_mapper, meta_node_component_ids, meta_node_keys, [KEY] * len(meta_node_keys)

    def meta_edge_info(self, meta_node_id_mapper: Dict[Tuple[int, str], int]) -> Tuple[
        List[int], List[int], List[int]
    ]:
        meta_edge_ids: List[int] = []
        meta_edge_id_mapper: Dict[Tuple[int, str, str], int] = {}
        src_meta_node_ids: List[int] = []
        dst_meta_node_ids: List[int] = []
        for src, dst in zip(self.src, self.dst):
            src_cpn_id, dst_cpn_id = self.component_id[src], self.component_id[dst]
            src_name, dst_name = self.key[src], self.key[dst]
            if (src_cpn_id, src_name, dst_name) not in meta_edge_id_mapper:
                meta_edge_id_mapper[(src_cpn_id, src_name, dst_name)] = len(meta_edge_id_mapper)
                src_meta_node_ids.append(meta_node_id_mapper[(src_cpn_id, src_name)])
                dst_meta_node_ids.append(meta_node_id_mapper[(dst_cpn_id, dst_name)])
            meta_edge_ids.append(meta_edge_id_mapper[(src_cpn_id, src_name, dst_name)])
        return meta_edge_ids, src_meta_node_ids, dst_meta_node_ids

    @staticmethod
    def segment_key(component_ids, keys, srcs, dsts, types):
        for i in range(len(keys)):
            segmented_key = jieba.lcut(keys[i])
            if len(segmented_key) > 1:
                srcs += list(range(len(keys), len(keys) + len(segmented_key)))
                dsts += [i] * len(segmented_key)  # 指向原本的点
                keys += segmented_key
                component_ids += [component_ids[i]] * len(segmented_key)
                types += [META] * len(segmented_key)

    @classmethod
    def from_data(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def reduce(cls, structs, **kwargs):
        raise NotImplementedError

    @classmethod
    def merge(cls, structs, **kwargs):
        raise NotImplementedError
