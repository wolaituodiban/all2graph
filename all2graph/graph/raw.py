from itertools import permutations
from typing import Dict, List, Union, Tuple

import numpy as np

from ..preserves import KEY, VALUE, TARGET, READOUT, META
from ..meta_struct import MetaStruct
from ..utils import Tokenizer


class RawGraph(MetaStruct):
    def __init__(self, component_id=None, key=None, value=None, src=None, dst=None, symbol=None):
        super().__init__(initialized=True)
        self.component_id: List[int] = list(component_id or [])
        self.key: List[str] = list(key or [])
        self.value: List[Union[Dict, List, str, int, float, None]] = list(value or [])
        self.src: List[int] = list(src or [])
        self.dst: List[int] = list(dst or [])
        self.symbol: List[str] = list(symbol or [])

        assert len(self.component_id) == len(self.key) == len(self.value) == len(self.symbol)
        assert len(self.src) == len(self.dst)

    def __eq__(self, other):
        return super().__eq__(other) \
               and self.component_id == other.component_id \
               and self.key == other.key \
               and self.value == other.value \
               and self.src == other.src \
               and self.dst == other.dst \
               and self.symbol == other.symbol

    def to_json(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_json(cls, obj: dict):
        raise NotImplementedError

    @property
    def num_nodes(self):
        return len(self.key)

    @property
    def num_edges(self):
        return len(self.src)

    @property
    def num_components(self):
        return np.unique(self.component_id).shape[0]

    @property
    def num_keys(self):
        return np.unique(self.key).shape[0]

    @property
    def num_types(self):
        return np.unique(self.symbol).shape[0]

    def insert_edges(self, srcs: List[int], dsts: List[int], bidirection=False):
        if bidirection:
            self.src += srcs + dsts
            self.dst += dsts + srcs
        else:
            self.src += srcs
            self.dst += dsts

    def insert_node(
            self,
            component_id: int,
            key: str,
            value: Union[dict, list, str, int, float, bool, None],
            self_loop: bool,
            symbol=VALUE
    ) -> int:
        """

        :param.py component_id:
        :param.py key:
        :param.py value:
        :param.py self_loop:
        :param.py symbol:
        :return:
        """
        node_id = len(self.key)
        self.component_id.append(component_id)
        self.key.append(key)
        self.value.append(value)
        self.symbol.append(symbol)
        if self_loop:
            self.src.append(node_id)
            self.dst.append(node_id)
        return node_id

    def insert_readout(
            self,
            component_id: int,
            value: Union[dict, list, str, int, float, bool, None],
            self_loop: bool
    ) -> int:
        return self.insert_node(component_id=component_id, key=READOUT, value=value, self_loop=self_loop, symbol=READOUT)

    def add_targets(self, targets):
        new_graph = RawGraph(component_id=list(self.component_id), key=list(self.key), value=list(self.value),
                             src=list(self.src), dst=list(self.dst), symbol=list(self.symbol))
        for i, type in enumerate(new_graph.symbol):
            if type == READOUT:
                for target in targets:
                    target_id = new_graph.insert_node(
                        new_graph.component_id[i], target, value=TARGET, self_loop=False, symbol=TARGET)
                    new_graph.insert_edges([i], [target_id])
        return new_graph

    @property
    def edge_key(self):
        return [(self.key[src], self.key[dst]) for src, dst in zip(self.src, self.dst)]

    def _meta_node_info(self) -> Tuple[
        List[int], List[str], Dict[Tuple[int, str], int], List[int]
    ]:
        meta_node_id: List[int] = []
        meta_node_id_mapper: Dict[Tuple[int, str], int] = {}
        meta_component_id: List[int] = []
        meta_value: List[str] = []
        for i, key in zip(self.component_id, self.key):
            if (i, key) not in meta_node_id_mapper:
                meta_node_id_mapper[(i, key)] = len(meta_node_id_mapper)
                meta_component_id.append(i)
                meta_value.append(key)
            meta_node_id.append(meta_node_id_mapper[(i, key)])
        return meta_component_id, meta_value, meta_node_id_mapper, meta_node_id

    def _meta_edge_info(self, meta_node_id_mapper: Dict[Tuple[int, str], int]) -> Tuple[
        List[int], List[int], List[int]
    ]:
        meta_edge_id: List[int] = []
        meta_edge_id_mapper: Dict[Tuple[int, str, str], int] = {}
        meta_src: List[int] = []
        meta_dst: List[int] = []
        for src, dst in zip(self.src, self.dst):
            src_cpn_id, dst_cpn_id = self.component_id[src], self.component_id[dst]
            src_name, dst_name = self.key[src], self.key[dst]
            if (src_cpn_id, src_name, dst_name) not in meta_edge_id_mapper:
                meta_edge_id_mapper[(src_cpn_id, src_name, dst_name)] = len(meta_edge_id_mapper)
                meta_src.append(meta_node_id_mapper[(src_cpn_id, src_name)])
                meta_dst.append(meta_node_id_mapper[(dst_cpn_id, dst_name)])
            meta_edge_id.append(meta_edge_id_mapper[(src_cpn_id, src_name, dst_name)])
        return meta_src, meta_dst, meta_edge_id

    def meta_graph(self, tokenizer: Tokenizer = None):
        """元图是以图的所有唯一的key为value组成的图，元图的key和元图的value是相同的"""
        meta_component_id, meta_value, meta_node_id_mapper, meta_node_id = self._meta_node_info()
        meta_src, meta_dst, meta_edge_id = self._meta_edge_info(meta_node_id_mapper)
        meta_type = [KEY] * len(meta_component_id)
        if tokenizer is not None:
            for i in range(len(meta_value)):
                segmented_key = tokenizer.lcut(meta_value[i])
                if len(segmented_key) > 1:
                    all_ids = [i] + list(range(len(meta_value), len(meta_value) + len(segmented_key)))
                    for src, dst in permutations(all_ids, 2):
                        meta_src.append(src)
                        meta_dst.append(dst)
                    meta_value += segmented_key
                    meta_component_id += [meta_component_id[i]] * len(segmented_key)
                    meta_type += [META] * len(segmented_key)
        meta_graph = RawGraph(
            component_id=meta_component_id, key=meta_value, value=meta_value, src=meta_src, dst=meta_dst, symbol=meta_type
        )
        return meta_graph, meta_node_id, meta_edge_id

    def to_df(self, *attrs):
        import pandas as pd
        node_df = pd.DataFrame({attr: getattr(self, attr) for attr in attrs})
        edge_df = pd.DataFrame({'src': self.src, 'dst': self.dst})
        for col, series in node_df.iteritems():
            edge_df['src_{}'.format(col)] = series[edge_df.src].values
            edge_df['dst_{}'.format(col)] = series[edge_df.dst].values
        return edge_df

    def drop_duplicated_edges(self):
        df = self.to_df().drop_duplicates()
        self.src = df.src.tolist()
        self.dst = df.dst.tolist()

    @classmethod
    def from_data(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def reduce(cls, structs, **kwargs):
        raise NotImplementedError

    @classmethod
    def merge(cls, structs, **kwargs):
        raise NotImplementedError

    def extra_repr(self) -> str:
        return 'num_nodes={}, num_edges={}, num_components={}, num_keys={}, num_types={}'.format(
            self.num_nodes, self.num_edges, self.num_components, self.num_keys, self.num_types
        )
