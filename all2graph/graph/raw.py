from itertools import permutations
from typing import Dict, List, Union, Tuple, Iterable, Set

import numpy as np

from ..preserves import KEY, VALUE, TARGET, READOUT, META
from ..meta_struct import MetaStruct
from ..utils import Tokenizer

# todo 支持点分类和点回归，支持mask自监督
# 考虑在此模式下，将symbol作为他用，比如用于存储点的label
# 或者增加设计labels，并设计symbol和labels启用的开关
# 考虑dataset的兼容性和factory的兼容性
# 考虑encoder的兼容性

class RawGraph(MetaStruct):
    def __init__(self, component_id=None, key=None, value=None, src=None, dst=None, symbol=None, initialized=True,
                 **kwargs):
        super().__init__(initialized=initialized, **kwargs)
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
            component_id=meta_component_id, key=meta_value, value=meta_value, src=meta_src, dst=meta_dst,
            symbol=meta_type)
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
    def batch(cls, graphs: Iterable):
        new_graph = cls()
        for graph in graphs:
            num_components = new_graph.num_components
            num_nodes = new_graph.num_nodes
            new_graph.component_id += [i + num_components for i in graph.component_id]
            new_graph.key += graph.key
            new_graph.value += graph.value
            new_graph.src += [i + num_nodes for i in graph.src]
            new_graph.dst += [i + num_nodes for i in graph.dst]
            new_graph.symbol += graph.symbol
        return new_graph

    def extra_repr(self) -> str:
        return 'num_nodes={}, num_edges={}, num_components={}, num_keys={}, num_types={}'.format(
            self.num_nodes, self.num_edges, self.num_components, self.num_keys, self.num_types
        )

    def to_json(self, drop_nested_value=True) -> dict:
        output = super().to_json()
        output['component_id'] = self.component_id
        output['key'] = self.key
        if drop_nested_value:
            output['value'] = [None if isinstance(x, (dict, list)) else x for x in self.value]
        else:
            output['value'] = self.value
        output['src'] = self.src
        output['dst'] = self.dst
        output['symbol'] = self.symbol
        return output

    @classmethod
    def from_json(cls, obj: dict):
        return super().from_json(obj)

    @classmethod
    def from_data(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def reduce(cls, structs, weights=None, **kwargs):
        raise NotImplementedError

    def add_mask(self, p):
        """
        对value添加mask，用于预训练
        返回的RawGraph是原来的浅拷贝
        会修改mask对应位置的symbol为TARGET
        Args:
            p: mask的概率

        Returns:
            raw_graph: mask了一部分value的RawGraph
            masked_value: 被mask掉的value
        """
        new_symbol = []
        new_value = []
        mask_value = []
        for i in range(self.num_nodes):
            if np.random.rand() < p:
                new_symbol.append(TARGET)
                new_value.append(None)
                mask_value.append(self.value[i])
            else:
                assert self.symbol[i] != TARGET, 'mask的target与原来存在的target冲突'
                new_symbol.append(self.symbol[i])
                new_value.append(self.value[i])
        new_graph = RawGraph(
            component_id=self.component_id, key=self.key, value=new_value, src=self.src, dst=self.dst,
            symbol=new_symbol)
        return new_graph, mask_value

    def filter_node(self, keys: Set[str]):
        """
        return sub graph with given node keys
        Args:
            keys: set of str

        Returns:
            raw_graph: new RawGraph
            dropped_keys: set of dropped keys
        """
        dropped_keys = set()
        selected_nodes = []
        for i, k in enumerate(self.key):
            if k in keys:
                selected_nodes.append(i)
            else:
                dropped_keys.add(k)

        if len(dropped_keys) == 0:
            return self, dropped_keys

        component_id = [self.component_id[i] for i in selected_nodes]
        key = [self.key[i] for i in selected_nodes]
        value = [self.value[i] for i in selected_nodes]
        symbol = [self.symbol[i] for i in selected_nodes]

        id_mapper = {old: new for new, old in enumerate(selected_nodes)}
        src = []
        dst = []
        for s, d in zip(self.src, self.dst):
            if s in id_mapper and d in id_mapper:
                src.append(id_mapper[s])
                dst.append(id_mapper[d])

        new_graph = RawGraph(component_id=component_id, key=key, value=value, symbol=symbol, src=src, dst=dst)
        return new_graph, dropped_keys

    def filter_edge(self, keys: Set[Tuple[str, str]]):
        """
        return sub graph with given edge keys
        Args:
            keys:

        Returns:
            new_graph: sub graph
            dropped_keys: set of dropped keys
        """
        dropped_keys: Set[Tuple[str, str]] = set()
        src = []
        dst = []
        for s, d in zip(self.src, self.dst):
            k = (self.key[s], self.key[d])
            if k not in keys:
                src.append(s)
                dst.append(d)
            else:
                dropped_keys.add(k)
        if len(dropped_keys) == 0:
            return self, dropped_keys
        else:
            new_graph = RawGraph(
                component_id=self.component_id, key=self.key, value=self.value, symbol=self.symbol, src=src, dst=dst)
            return new_graph, dropped_keys
