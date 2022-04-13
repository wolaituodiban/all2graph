import gzip
import pickle
from itertools import combinations
from typing import List, Set, Dict, Tuple

import numpy as np
import pandas as pd

from ..globals import *
from ..meta_struct import MetaStruct
from ..utils import tqdm


class SeqInfo(MetaStruct):
    def __init__(self,
                 seq_mapper: Dict[Tuple[int, str], int],
                 seq_sample: List[int],
                 seq_type: List[str],
                 type2node: Dict[str, List[int]],
                 seq2node: List[Tuple[int, int]],
                 **kwargs):
        super().__init__(initialized=True, **kwargs)
        self.seq_sample = seq_mapper
        self.seq_type = seq_type
        self.seq_sample = seq_sample
        self.type2node = type2node
        self.seq2node = seq2node


class RawGraph(MetaStruct):
    def __init__(self, **kwargs):
        super().__init__(initialized=True, **kwargs)
        # sequence中每个元素对应的图上的node
        # 每个seq都由(sample, type)二元组进行索引
        # 每个元素是一个list，包含node
        self.seq_len: Dict[Tuple[int, str], int] = {}
        # 图上每个node对应的(sample, type, loc)二元组
        self.nodes: List[Tuple[int, str, int]] = []
        # 图上的node对应的value
        self.values = []
        self.edges = [], []
        self.targets = set()

        # 记录作为要被当作外键的node
        # {sample: {type: {value: node}}}
        self._local_foreign_keys: Dict[int, Dict[str, Dict[str, int]]] = {}
        # {type: {value: node}}
        self._global_foreign_keys: Dict[str, Dict[str, int]] = {}

    def _assert(self):
        assert len(self.nodes) == len(self.values) == sum(self.seq_len.values())
        assert len(self.edges[0]) == len(self.edges[1])

    @property
    def num_types(self):
        return len(self.unique_types)

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
    def samples(self):
        return [x[0] for x in self.nodes]

    @property
    def types(self):
        return [x[1] for x in self.nodes]

    @property
    def num_samples(self):
        return len(set(x[0] for x in self.nodes))

    @property
    def max_seq_len(self):
        return max(self.seq_len.values())

    @property
    def unique_types(self) -> Set[str]:
        return set(key[1] for key in self.seq_len).union(self.targets)

    @property
    def foreign_key_types(self) -> Set[str]:
        foreign_key_types = list(self._global_foreign_keys)
        for t in self._local_foreign_keys.values():
            foreign_key_types.extend(t)
        return set(foreign_key_types)

    @property
    def formatted_values(self) -> Tuple[np.ndarray, np.ndarray]:
        id_keys = self.foreign_key_types
        values = []
        for key, value in zip(self.types, self.values):
            if key in id_keys or not isinstance(value, (str, float, int, bool)):
                values.append(None)
            else:
                values.append(value)

        numbers = pd.to_numeric(values, errors='coerce')
        mask = np.bitwise_not(np.isnan(numbers))
        strings = np.array(values, dtype=object)
        strings[mask] = None
        return strings, numbers

    @property
    def seq(self) -> Dict[Tuple[int, str], List[int]]:
        seq = {}
        for sample, t, loc in self.nodes:
            if (sample, t) not in seq:
                seq[sample, t] = [loc]
            else:
                seq[sample, t].append(loc)
        return seq

    @property
    def seq_info(self) -> SeqInfo:
        # parsing seq2node
        seq_mapper = {}
        seq_type = []
        seq_sample = []
        type2node = {}

        for (sample, t), nodes in self.seq_len.items():
            seq_mapper[(sample, t)] = len(seq_mapper)
            seq_type.append(t)
            seq_sample.append(sample)
            if t not in type2node:
                type2node[t] = []

        # parsing node2seq
        seq2node = []
        for i, (sample, t, loc) in enumerate(self.nodes):
            seq2node.append((seq_mapper[sample, t], loc))
            type2node[t].append(i)
        return SeqInfo(
            seq_mapper=seq_mapper, seq_type=seq_type, seq_sample=seq_sample, type2node=type2node,
            seq2node=seq2node)

    @property
    def node_df(self) -> pd.DataFrame:
        strings, numbers = self.formatted_values
        df = pd.DataFrame({SAMPLE: self.samples, TYPE: self.types, STRING: strings, NUMBER: numbers})
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

    def add_kv_(self, sample: int, key: str, value) -> int:
        """返回新增的entity的id"""
        vid = len(self.values)
        if (sample, key) not in self.seq_len:
            self.seq_len[sample, key] = 1
        else:
            self.seq_len[sample, key] += 1
        self.nodes.append((sample, key, self.seq_len[sample, key] - 1))
        self.values.append(value)
        return vid

    def add_targets_(self, keys: Set[str]):
        assert self.unique_types.isdisjoint(keys), '{} already exists'.format(self.unique_types.intersection(keys))
        self.targets = self.targets.union(keys)

    def __add_foreign_key_(self, sample, key, value, fk_dict):
        if key not in fk_dict:
            fk_dict[key] = {}
        fk_dict = fk_dict[key]
        if value not in fk_dict:
            vid = self.add_kv_(sample, key, value)
            fk_dict[value] = vid
        return fk_dict[value]

    def add_local_foreign_key_(self, sample, key: str, value: str) -> int:
        """如果id已存在，则不会对图产生任何修改"""
        if sample not in self._local_foreign_keys:
            self._local_foreign_keys[sample] = {}
        return self.__add_foreign_key_(sample, key, value, self._local_foreign_keys[sample])

    def add_global_foreign_key_(self, sample, key: str, value: str) -> int:
        """如果id已存在，则不会对图产生任何修改"""
        return self.__add_foreign_key_(sample, key, value, self._global_foreign_keys)

    def to_networkx(self, exclude_types: Set[str] = None, include_types: Set[str] = None, disable=True):
        include_types = include_types or self.unique_types
        if exclude_types:
            include_types = include_types.difference(exclude_types)
        from networkx import MultiDiGraph
        graph = MultiDiGraph()
        for u, v in tqdm(zip(*self.edges), disable=disable, postfix='add edges'):
            graph.add_edge(u, v)

        for i, ((sample, t, _), value) in tqdm(
                enumerate(zip(self.nodes, self.values)), disable=disable, postfix='add nodes'):
            if t in include_types:
                graph.add_node(i, **{SAMPLE: sample, TYPE: t, VALUE: value})
            else:
                graph.remove_nodes_from(i)
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
        graph = self.to_networkx(include_types=include_keys, exclude_types=exclude_keys, disable=disable)

        # 位置
        if pos == 'planar':
            pos = planar_layout(graph, scale=scale, center=center, dim=dim)
        # 设置颜色
        types = [attr[TYPE] for node, attr in graph.nodes.items()]
        node_color = pd.factorize(types)[0]
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
        for i, t in tqdm(enumerate(types), disable=disable, desc='add legends'):
            if t in patches:
                continue
            patches[t] = mpatches.Patch(color=node_color[i], label=t)
        ax.legend(handles=patches.values())

        return fig, ax

    def extra_repr(self) -> str:
        return 'num_samples={}, num_keys={}, num_nodes={}, num_edges={}, num_targets={}'.format(
            self.num_samples, self.num_types, self.num_nodes, self.num_edges, self.num_targets
        )

    @classmethod
    def load(cls, path, **kwargs):
        with gzip.open(path, 'rb', **kwargs) as file:
            return pickle.load(file)

    def save(self, path, **kwargs):
        with gzip.open(path, 'wb', **kwargs) as file:
            pickle.dump(self, file)
