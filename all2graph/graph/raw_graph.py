import gzip
import pickle
from itertools import combinations
from typing import List, Set, Dict

import numpy as np
import pandas as pd

from ..globals import *
from ..meta_struct import MetaStruct
from ..utils import tqdm


class RawGraph(MetaStruct):
    def __init__(self, **kwargs):
        super().__init__(initialized=True, **kwargs)
        self.nodes_per_sample = []
        self.keys = []
        self.values = []
        self.edges = [], []
        self.targets = set()

        # 记录作为要被当作id的value的id
        self._lids: List[Dict[str, Dict[str, int]]] = [{}]  # {sid: {value: vid}}
        self._gids: Dict[str, Dict[str, int]] = {}  # {value: vid}

    def _assert(self):
        assert sum(self.nodes_per_sample) == len(self.keys) == len(self.values)
        assert len(self.edges[0]) == len(self.edges[1])
        if len(self.edges[0]) > 0:
            assert len(self.keys) >= max(self.edges[1] + self.edges[0])

    @property
    def unique_keys(self):
        return set(self.keys).union(self.targets)

    @property
    def id_keys(self):
        id_keys = list(self._gids)
        for lids in self._lids:
            id_keys.extend(lids)
        return set(id_keys)

    @property
    def formatted_values(self):
        id_keys = self.id_keys
        values = []
        for key, value in zip(self.keys, self.values):
            if key in id_keys or not isinstance(value, (str, float, int, bool)):
                values.append(None)
            else:
                values.append(value)
        return values

    @property
    def num_samples(self):
        return len(self.nodes_per_sample)

    @property
    def num_keys(self):
        return len(self.unique_keys)

    @property
    def num_nodes(self):
        return len(self.values)

    @property
    def num_edges(self):
        return len(self.edges[0])

    @property
    def num_targets(self):
        return len(self.targets)

    @property
    def node_df(self) -> pd.DataFrame:
        df = pd.DataFrame({SAMPLE: None, KEY: self.keys, STRING: self.formatted_values})
        indices = np.cumsum([0] + self.nodes_per_sample)
        for k, (i, j) in enumerate(zip(indices[:-1], indices[1:])):
            df.loc[i:j, SAMPLE] = k
        df[NUMBER] = pd.to_numeric(df[STRING], errors='coerce')
        df.loc[df[NUMBER].notna(), STRING] = None
        return df

    def add_edge_(self, u, v):
        self.edges[0].append(u)
        self.edges[1].append(v)

    def add_edges_(self, u: List[int], v: List[int]):
        self.edges[0].extend(u)
        self.edges[1].extend(v)

    def add_dense_edges_(self, x):
        for u, v in combinations(x, 2):
            self.edges[0].append(u)
            self.edges[0].append(v)
            self.edges[1].append(v)
            self.edges[1].append(u)

    def add_split_(self):
        self._lids.append({})
        if len(self.nodes_per_sample) == 0:
            self.nodes_per_sample.append(self.num_nodes)
        else:
            self.nodes_per_sample.append(self.num_nodes - sum(self.nodes_per_sample))

    def add_kv_(self, key: str, value) -> int:
        """返回新增的entity的id"""
        vid = len(self.values)
        self.values.append(value)
        self.keys.append(key)
        return vid

    def add_targets_(self, keys: Set[str]):
        assert self.unique_keys.isdisjoint(keys), '{} already exists'.format(self.unique_keys.intersection(keys))
        self.targets = self.targets.union(keys)

    def __add_id_(self, key, value, ids):
        if key not in ids:
            ids[key] = {}
        ids = ids[key]
        if value not in ids:
            vid = self.add_kv_(key, value)
            ids[value] = vid
        return ids[value]

    def add_lid_(self, key: str, value: str) -> int:
        """如果id已存在，则不会对图产生任何修改"""
        return self.__add_id_(key, value, self._lids[-1])

    def add_gid_(self, key: str, value: str) -> int:
        """如果id已存在，则不会对图产生任何修改"""
        return self.__add_id_(key, value, self._gids)

    def to_networkx(self, exclude_keys: Set[str] = None, include_keys: Set[str] = None, disable=True):
        include_keys = include_keys or self.unique_keys
        if exclude_keys:
            include_keys = include_keys.difference(exclude_keys)
        from networkx import MultiDiGraph
        graph = MultiDiGraph()
        for u, v in tqdm(zip(*self.edges), disable=disable, postfix='add edges'):
            graph.add_edge(u, v)

        i = 0  # sample计数器
        for j, (key, value) in tqdm(enumerate(zip(self.keys, self.values)), disable=disable, postfix='add nodes'):
            if sum(self.nodes_per_sample[:i + 1]) <= j:
                i += 1
            if key in include_keys:
                graph.add_node(j, **{SAMPLE: i, KEY: key, VALUE: value})
            else:
                graph.remove_nodes_from(j)
        return graph

    def draw(
            self, include_keys=None, exclude_keys=None, disable=True, pos='planar', scale=1, center=None,
            dim=2, norm=None, cmap='rainbow', ax=None, labels=None, with_labels=True, **kwargs
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
            norm: 详情间network.draw
            cmap: 详情间network.draw
            ax: matplotlib Axes object
            labels: 详情间network.draw
            with_labels: 详情间network.draw
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
        graph = self.to_networkx(include_keys=include_keys, exclude_keys=exclude_keys, disable=disable)

        # 位置
        if pos == 'planar':
            pos = planar_layout(graph, scale=scale, center=center, dim=dim)
        # 设置颜色
        keys = [attr[KEY] for node, attr in graph.nodes.items()]
        node_color = pd.factorize(keys)[0]
        node_color = ScalarMappable(norm=norm, cmap=cmap).to_rgba(node_color)

        def format_value(value):
            if isinstance(value, dict):
                return '{}'
            elif isinstance(value, list):
                return '[]'
            else:
                return value

        if labels is None:
            labels = {node: format_value(attr[VALUE]) for node, attr in graph.nodes.items()}
        draw(graph, pos=pos, ax=ax, node_color=np.array(node_color), labels=labels, with_labels=with_labels, **kwargs)

        # 加标注
        patches = {}
        for i, key in tqdm(enumerate(keys), disable=disable, desc='add legends'):
            if key in patches:
                continue
            patches[key] = mpatches.Patch(color=node_color[i], label=key)
        ax.legend(handles=patches.values())

        return fig, ax

    def extra_repr(self) -> str:
        return 'num_samples={}, num_keys={}, num_nodes={}, num_edges={}, num_targets={}'.format(
            self.num_samples, self.num_keys, self.num_nodes, self.num_edges, self.num_targets
        )

    @classmethod
    def load(cls, path, **kwargs):
        with gzip.open(path, 'rb', **kwargs) as file:
            return pickle.load(file)

    def save(self, path, **kwargs):
        with gzip.open(path, 'wb', **kwargs) as file:
            pickle.dump(self, file)
