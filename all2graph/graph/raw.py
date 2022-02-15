from itertools import permutations
from typing import Dict, List, Union, Tuple, Iterable, Set

import numpy as np
import pandas as pd

from ..preserves import KEY, VALUE, TARGET, READOUT, META
from ..meta_struct import MetaStruct
from ..utils import Tokenizer, tqdm


def sequence_edge(node_ids: List[int], degree: int, r_degree: int) -> Tuple[List[int], List[int]]:
    new_dsts = []
    new_srcs = []
    for i, node_id in enumerate(node_ids):
        # 正向
        end = i + degree + 1 if degree >= 0 else len(node_ids)
        new_dsts += node_ids[i + 1:end]
        # 反向
        start = max(0, i - r_degree) if r_degree >= 0 else 0
        new_dsts += node_ids[start:i]
        # 补全
        new_srcs += [node_id] * (len(new_dsts) - len(new_srcs))
    return new_dsts, new_srcs


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

    def copy(self):
        return RawGraph(component_id=list(self.component_id), key=list(self.key), value=list(self.value),
                 src=list(self.src), dst=list(self.dst), symbol=list(self.symbol))

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
        return self.insert_node(
            component_id=component_id, key=READOUT, value=value, self_loop=self_loop, symbol=READOUT)

    def add_targets(self, targets, inplace=False):
        if inplace:
            new_graph = self
        else:
            new_graph = self.copy()
        for i, _type in enumerate(new_graph.symbol):
            if _type == READOUT:
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

    def add_mask(self, p, inplace=False):
        """
        对value添加mask，用于预训练
        返回的RawGraph是原来的浅拷贝
        会修改mask对应位置的symbol为TARGET
        Args:
            p: mask的概率
            inplace: 如果是，按么在原来的图上做修改
        Returns:
            raw_graph: mask了一部分value的RawGraph
            masked_value: 被mask掉的value
        """
        if inplace:
            new_graph = self
        else:
            new_graph = self.copy()
        mask_value = []
        for i in range(self.num_nodes):
            if np.random.rand() < p:
                new_graph.symbol[i] = TARGET
                new_graph.value[i] = None
                mask_value.append(self.value[i])
        return new_graph, mask_value

    def filter_node(self, keys: Union[dict, Set[str]]):
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
            if k in keys:
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

    def to_networkx(self, disable=True, exclude_keys=None, include_keys=None):
        import networkx as nx

        nx_graph = nx.MultiDiGraph()
        for i, (cid, key, value, symbol) in tqdm(
                enumerate(zip(self.component_id, self.key, self.value, self.symbol)), disable=disable, desc='add nodes'
        ):
            if exclude_keys is not None and key in exclude_keys:
                continue
            if include_keys is None or key in include_keys:
                nx_graph.add_node(i, component_id=cid, key=key, value=value, symbol=symbol)
        for src, dst in tqdm(zip(self.src, self.dst), disable=disable, desc='add edges'):
            src_key = self.key[src]
            dst_key = self.key[dst]
            if exclude_keys is not None and (src_key in exclude_keys or dst_key in exclude_keys):
                continue
            if include_keys is None or (src_key in include_keys and dst_key in include_keys):
                nx_graph.add_edge(src, dst, key=(src_key, dst_key))

        return nx_graph

    def draw(
            self, include_keys=None, exclude_keys=None, disable=True, pos='planar', scale=1, center=None, dim=2,
            node_size=32, arrowsize=8, norm=None, cmap='nipy_spectral', with_labels=False, ax=None,
            **kwargs
    ):
        """

        Args:
            include_keys: 仅包含key对应的点
            exclude_keys: 去掉key对应的点
            disable: 禁用进度条
            pos: 图中每个点的坐标，默认会使用network.planar_layout计算最优坐标
            scale: 详情间network.planar_layout
            center: 详情间network.planar_layout
            dim: 详情间network.planar_layout
            node_size: 点的大小
            arrowsize: 箭头大小
            norm: 详情间network.draw
            cmap: 详情间network.draw
            with_labels: 详情间network.draw
            ax: matplotlib Axes object
            **kwargs: 详情间network.draw

        Returns:

        """
        # todo planar list子节点位置修正  list和dict边颜色区分
        import matplotlib.pyplot as plt
        from networkx.drawing import draw
        from networkx.drawing.layout import planar_layout
        import matplotlib.patches as mpatches
        from matplotlib.cm import ScalarMappable

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        # 转成networkx
        nx_graph = self.to_networkx(include_keys=include_keys, exclude_keys=exclude_keys, disable=disable)
        labels = {node: attr['key'] for node, attr in nx_graph.nodes.items()}
        if pos == 'planar':
            pos = planar_layout(nx_graph, scale=scale, center=center, dim=dim)

        # 设置颜色
        node_color = pd.factorize(list(labels.values()))[0]
        node_color = ScalarMappable(norm=norm, cmap=cmap).to_rgba(node_color)

        # 画
        draw(
            nx_graph, pos=pos, ax=ax, node_size=node_size, arrowsize=arrowsize, node_color=np.array(node_color),
            labels=labels, with_labels=with_labels, **kwargs
        )

        # 加标注
        patches = {}
        for i, key in tqdm(enumerate(labels.values()), disable=disable, desc='add legends'):
            if key in patches:
                continue
            patches[key] = mpatches.Patch(color=node_color[i], label=key)
        ax.legend(handles=patches.values())

        return fig, ax

    def add_key_edge(self, degree, r_degree, inplace=False):
        """
        在key相同当点之间增加边
        Args:
            degree: 正向的度数
            r_degree: 反向的度数
            inplace: 如果False，那么不会改变原来的图，并且返回一个修改了的副本

        Returns:

        """
        if inplace:
            graph = self
        else:
            graph = self.copy()

        new_dsts = []
        new_srcs = []
        df = pd.DataFrame({'componet_id': self.component_id, 'key': self.key, 'value': self.value})
        df = df[df.value.apply(lambda x: not isinstance(x, dict) and not isinstance(x, list))]
        for _, group in df.groupby(['componet_id', 'key']):
            new_dsts2, new_srcs2 = sequence_edge(group.index.tolist(), degree=degree, r_degree=r_degree)
            new_dsts += new_dsts2
            new_srcs += new_srcs2
        graph.insert_edges(dsts=new_dsts, srcs=new_srcs)
        return graph
