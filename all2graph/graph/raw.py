from typing import List, Union, Tuple

import numpy as np
import pandas as pd

from ..meta_struct import MetaStruct
from ..utils import tqdm
from ..globals import *


class RawGraph(MetaStruct):
    """
    异构图
    有三种类型的点：
        key：代表entity或者target的类型，拥有一个属性
            value：表示key本身的值
        value：代表实体，拥用两个属性
            sample id：表示样本编号
            value：表示entity本身的值
        target：代表目标
    有五种类型的边：
        key2key：多对多
        key2value：一对一
        key2target：一对一
        value2value：多对多
        value2target：一对多
    有一个graph level的属性：
        readout：记录所有是readout的value的id，所有与target相连的value都是readout，但是readout不一定与target相连
                 每一个sample的readout必须是唯一的
    """
    def __init__(self, **kwargs):
        super().__init__(initialized=True, **kwargs)
        self.keys = []
        self.sids = []  # sample id
        self.values = []
        self.edges = {
            KEY2KEY: [[], []],
            KEY2VALUE: [[], []],
            KEY2TARGET: [[], []],
            VALUE2VALUE: [[], []],
            VALUE2TARGET: [[], []]
        }
        self.__sid2rid = {}  # sample id和readout id的映射关系

        # key的id的索引，用于加快key图的自动创建机制
        self.__ori_kids = {}

        # 记录作为要被当作id的value的id
        self.__lids = {}  # {sid: {value: vid}}
        self.__gids = {}  # {value: vid}

    def _assert(self):
        assert len(self.sids) == len(self.values) == len(self.key2value[0]) == len(self.key2value[1]), (
                len(self.sids), len(self.values), len(self.key2value[0]), len(self.key2value[1])
        )
        assert len(self.key2target[0]) == len(self.key2target[1]) \
               == len(self.value2target[0]) == len(self.value2target[1]), (
            len(self.key2target[0]), len(self.key2target[1]), len(self.value2target[0]), len(self.value2target[1])
        )

    @property
    def num_samples(self):
        return len(set(self.sids))

    @property
    def key2key(self):
        return self.edges[KEY2KEY]

    @property
    def key2value(self):
        return self.edges[KEY2VALUE]

    @property
    def key2target(self):
        return self.edges[KEY2TARGET]

    @property
    def value2value(self):
        return self.edges[VALUE2VALUE]

    @property
    def value2target(self):
        return self.edges[VALUE2TARGET]

    def __eq__(self, other):
        return super().__eq__(other) \
               and self.keys == other.keys \
               and self.sids == self.sids \
               and self.values == other.values \
               and self.key2key == other.key2key \
               and self.key2value == other.key2value \
               and self.key2target == other.key2target \
               and self.value2value == other.value2value \
               and self.value2target == other.value2target \
               and self.__sid2rid == other.__sid2rid \
               and self.__ori_kids == other.__ori_kids \
               and self.__lids == other.__lids \
               and self.__gids == other.__gids

    def num_nodes(self, ntype=None):
        if ntype == KEY:
            return len(self.keys)
        elif ntype == VALUE:
            return len(self.sids)
        elif ntype == TARGET:
            return len(self.key2target[0])
        elif ntype is None:
            return self.num_nodes(KEY) + self.num_nodes(VALUE) + self.num_nodes(TARGET)
        else:
            raise ValueError('unknown ntype "{}", must be one of {}'.format(ntype, [KEY, VALUE, TARGET]))

    def num_edges(self, etype=None):
        if etype is None:
            return sum(map(self.num_edges, self.edges))
        elif etype in self.edges:
            return len(self.edges[etype][0])
        else:
            raise ValueError('unknown etype "{}", must be one of {}'.format(etype, self.edges.keys()))

    def get_keys(self, nids, ntype):
        if ntype == KEY:
            return [self.keys[i] for i in nids]
        elif ntype == VALUE:
            entity2key = {v: u for u, v in zip(*self.key2value)}
            return [self.keys[entity2key[i]] for i in nids]
        elif ntype == TARGET:
            target2key = {v: u for u, v in zip(*self.key2target)}
            return [self.keys[target2key[i]] for i in nids]
        else:
            raise ValueError('unknown ntype "{}", must be one of {}'.format(ntype, [KEY, VALUE, TARGET]))

    def get_values(self, nids):
        return [self.values[i] for i in nids]

    def get_sids(self, nids):
        return [self.sids[i] for i in nids]

    def _add_edge_(self, u, v, etype, bidirectional=False):
        self.edges[etype][0].append(u)
        self.edges[etype][1].append(v)
        if bidirectional:
            self.edges[etype][0].append(v)
            self.edges[etype][1].append(u)

    def _add_edges_(self, u: List[int], v: List[int], etype, bidirectional=False):
        if bidirectional:
            u, v = u + v, v + u
        self.edges[etype][0] += u
        self.edges[etype][1] += v

    # todo 考虑一些操作是否在Graph处做比较快
    def _add_edges_for_seq_(self, nids: List[int], etype, degree: int = -1, r_degree: int = -1):
        """
        为一些列点之间添加边
        Args:
            nids: 点坐标
            etype: 边类型
            degree: 正向度数，-1表示全部，0表示无
            r_degree: 反向度数，-1表示全部，0表示无

        Returns:

        """
        for i, nid in enumerate(nids):
            # 正向
            end = i + degree + 1 if degree >= 0 else len(nids)
            self.edges[etype][1] += nids[i + 1:end]
            # 反向
            start = max(0, i - r_degree) if r_degree >= 0 else 0
            self.edges[etype][1] += nids[start:i]
            # 补全
            self.edges[etype][0] += [nid] * (len(self.edges[etype][1]) - len(self.edges[etype][0]))

    def add_edge_(self, *args, **kwargs):
        self._add_edge_(*args, etype=VALUE2VALUE, **kwargs)

    def add_edges_(self, *args, **kwargs):
        self._add_edges_(*args, etype=VALUE2VALUE, **kwargs)

    def add_edges_for_seq_(self, *args, **kwargs):
        self._add_edges_for_seq_(*args, etype=VALUE2VALUE, **kwargs)

    def add_edges_for_seq_by_key_(self, keys=None, **kwargs):
        """
        为某一个key的所有value点组成点序列增加边
        Args:
            keys:
            **kwargs: add_edges_for_seq的参数

        Returns:

        """
        if keys is None:
            keys = self.__ori_kids
        if not isinstance(keys, list):
            keys = list(keys)
        for key in keys:
            kid = self.__ori_kids[key]
            vids = [v for u, v in zip(*self.edges[KEY2VALUE]) if u == kid]
            self._add_edges_for_seq_(vids, VALUE2VALUE, **kwargs)

    def __add_k_(self, key) -> int:
        """
        # key图的自动创建机制
        # 每当一个key-value pair被插入时，首先检查是否已有相同的key存在，
        # 如果没有否则则插入一个新点key的节点
        # 另外如果key是一个tuple，那么tuple中的每个元素都会单独插入一个点，并且插入边使这些点之间完全联通
        Args:
            key:

        Returns:
            key的坐标
        """
        if key not in self.__ori_kids:
            self.__ori_kids[key] = self.num_nodes(KEY)
            self.keys.append(key)
            if isinstance(key, tuple):
                self.keys += list(key)
                self._add_edges_for_seq_(list(range(self.__ori_kids[key], len(self.keys))), KEY2KEY)
        return self.__ori_kids[key]

    def __add_v_(self, sid, value) -> int:
        """
        # 自动添加readout机制
        # 每一个sid的第一个点会被作为readout
        Args:
            sid:
            value:

        Returns:

        """
        if isinstance(value, dict):
            value = {}
        elif isinstance(value, list):
            value = []

        vid = len(self.values)
        if sid not in self.__sid2rid:
            self.__sid2rid[sid] = vid
        self.sids.append(sid)
        self.values.append(value)
        return vid

    def __add_lv_(self, sid, value) -> Tuple[int, bool]:
        """
        加入local id，自动判断是否存在
        Args:
            sid:
            value:

        Returns:
            vid: 坐标
            flag: 是否新增
        """

        if sid not in self.__lids:
            self.__lids[sid] = {}
        lids = self.__lids[sid]
        flag = False
        if value not in lids:
            lids[value] = self.__add_v_(sid, value)
            flag = True
        return lids[value], flag

    def __add_gv_(self, value) -> Tuple[int, bool]:
        """
        加入global id，自动判断是否存在
        Args:
            value:

        Returns:
            vid: 坐标
            flag: 是否新增
        """
        flag = False
        if value not in self.__gids:
            self.__gids[value] = self.__add_v_(None, value)
            flag = True
        return self.__gids[value], flag

    def add_kv_(self, sid: int, key: Union[str, Tuple], value, self_loop: bool) -> int:
        """返回新增的entity的id"""
        kid = self.__add_k_(key)
        vid = self.__add_v_(sid, value)
        self._add_edge_(kid, vid, KEY2VALUE)
        if self_loop:
            self._add_edge_(vid, vid, VALUE2VALUE)
        return vid

    def add_targets_(self, keys: List[Union[str, Tuple]]):
        for kid in map(self.__add_k_, keys):
            num_targets = self.num_nodes(TARGET)
            tids = list(range(num_targets, num_targets+len(self.__sid2rid)))
            self._add_edges_([kid] * len(tids), tids, KEY2TARGET)
            self._add_edges_(list(self.__sid2rid), tids, VALUE2TARGET)

    def add_lid_(self, sid: int, key: Union[str, Tuple[str]], value: str, self_loop: bool) -> int:
        """如果id已存在，则不会对图产生任何修改"""
        kid = self.__add_k_(key)
        vid, flag = self.__add_lv_(sid, value)
        if flag:
            self._add_edge_(kid, vid, KEY2VALUE)
            if self_loop:
                self._add_edge_(vid, vid, VALUE2VALUE)
        return vid

    def add_gid_(self, key: Union[str, Tuple[str]], value: str, self_loop: bool) -> int:
        """如果id已存在，则不会对图产生任何修改"""
        kid = self.__add_k_(key)
        vid, flag = self.__add_gv_(value)
        if flag:
            self._add_edge_(kid, vid, KEY2VALUE)
            if self_loop:
                self._add_edge_(vid, vid, VALUE2VALUE)
        return vid

    def to_simple_(self, etype=None):
        """转换成没有平行边的简单图"""
        if etype is None:
            list(map(self.to_simple_, self.edges))
        elif etype in self.edges:
            us, vs = [], []
            for u, v in set(zip(*self.edges[etype])):
                us.append(u)
                vs.append(v)
            self.edges[etype] = [us, vs]
        else:
            raise ValueError('unknown etype "{}", must be one of {}'.format(etype, self.edges.keys()))

    def to_df(self, key=False, exclude_keys=None, include_keys=None):
        sids, uids, vids = [], [], []
        utypes, u_keys, u_values = [], [], []
        vtypes, v_keys, v_values = [], [], []

        # value graph
        u, v = self.edges[VALUE2VALUE]
        sids += self.get_sids(u)
        uids += u
        vids += v
        utypes += [VALUE] * len(u)
        vtypes += [VALUE] * len(v)
        u_keys += self.get_keys(u, VALUE)
        v_keys += self.get_keys(v, VALUE)
        u_values += self.get_values(u)
        v_values += self.get_values(v)

        # value2target
        u, v = self.edges[VALUE2TARGET]
        sids += [None] * len(u)
        uids += u
        vids += v
        utypes += [VALUE] * len(u)
        vtypes += [TARGET] * len(v)
        u_keys += self.get_keys(u, VALUE)
        v_keys += self.get_keys(v, TARGET)
        u_values += self.get_values(u)
        v_values += [None] * len(v)

        if key:
            # key graph
            u, v = self.edges[KEY2KEY]
            sids += [None] * len(u)
            uids += u
            vids += v
            utypes += [KEY] * len(u)
            vtypes += [KEY] * len(v)
            u_keys += [KEY] * len(u)
            v_keys += [KEY] * len(v)
            u_values += self.get_keys(u, KEY)
            v_values += self.get_keys(v, KEY)

            # key2value
            u, v = self.edges[KEY2VALUE]
            sids += self.get_sids(v)
            uids += u
            vids += v
            utypes += [KEY] * len(u)
            vtypes += [VALUE] * len(v)
            u_keys += [KEY] * len(u)
            v_keys += self.get_keys(v, VALUE)
            u_values += self.get_keys(u, KEY)
            v_values += self.get_values(v)

            # key2target
            u, v = self.edges[KEY2TARGET]
            sids += [None] * len(u)
            uids += u
            vids += v
            utypes += [KEY] * len(u)
            vtypes += [TARGET] * len(v)
            u_keys += [KEY] * len(u)
            v_keys += self.get_keys(v, TARGET)
            u_values += self.get_keys(u, KEY)
            v_values += [None] * len(v)

        df = pd.DataFrame({SID: sids, 'u': uids,  'utype': utypes, 'u_key': u_keys, 'u_value': u_values,
                           'vtype': vtypes, 'v': vids, 'v_key': v_keys,  'v_value': v_values})
        if exclude_keys is not None:
            df['exclude_mask'] = False
            df['exclude_mask'] += (df.utype == KEY) & (df.u_value.isin(exclude_keys))
            df['exclude_mask'] += (df.vtype == KEY) & (df.v_value.isin(exclude_keys))
            df['exclude_mask'] += (df.utype != KEY) & (df.u_key.isin(exclude_keys))
            df['exclude_mask'] += (df.vtype != KEY) & (df.v_key.isin(exclude_keys))
            df = df.drop(index=df['exclude_mask'])
            df = df.drop(columns='exclude_mask')
        if include_keys is not None:
            df['include_mask'] = False
            df['include_mask'] += (df.utype == KEY) & (df.u_value.isin(include_keys))
            df['include_mask'] += (df.vtype == KEY) & (df.v_value.isin(include_keys))
            df['include_mask'] += (df.utype != KEY) & (df.u_key.isin(include_keys))
            df['include_mask'] += (df.vtype != KEY) & (df.v_key.isin(include_keys))
            df = df.drop(index=df['include_mask'])
            df = df.drop(columns='include_mask')
        return df

    def to_networkx(self, key=False, exclude_keys=None, include_keys=None, disable=True):
        from networkx import MultiDiGraph
        graph = MultiDiGraph()
        df = self.to_df(key=key, exclude_keys=exclude_keys, include_keys=include_keys)
        for _, row in tqdm(df.iterrows(), disable=disable):
            u = row.utype + str(row.u)
            v = row.vtype + str(row.v)
            graph.add_edge(u, v)
            graph.add_node(u, **{SID: row.sid, KEY: row.u_key, VALUE: row.u_value})
            graph.add_node(v, **{SID: row.sid, KEY: row.v_key, VALUE: row.v_value})
        return graph

    def draw(
            self, key=False, include_keys=None, exclude_keys=None, disable=True, pos='planar', scale=1, center=None,
            dim=2, norm=None, cmap='rainbow', ax=None, labels=None, with_labels=True, **kwargs
    ):
        """

        Args:
            key: 是否包含key graph
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
        graph = self.to_networkx(key=key, include_keys=include_keys, exclude_keys=exclude_keys, disable=disable)

        # 位置
        if pos == 'planar':
            pos = planar_layout(graph, scale=scale, center=center, dim=dim)
        # 设置颜色
        keys = [attr[KEY] for node, attr in graph.nodes.items()]
        node_color = pd.factorize(keys)[0]
        node_color = ScalarMappable(norm=norm, cmap=cmap).to_rgba(node_color)

        if labels is None:
            labels = {node: attr[VALUE] for node, attr in graph.nodes.items()}
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
        return 'num_nodes={},\nnum_edges={}'.format(
            json.dumps({ntype: self.num_nodes(ntype) for ntype in [KEY, VALUE, TARGET]}, indent=1),
            json.dumps({str(etype): self.num_edges(etype) for etype in self.edges}, indent=1)
        )

    def to_json(self, drop_nested_value=True) -> dict:
        raise NotImplementedError

    @classmethod
    def from_json(cls, obj: dict):
        return super().from_json(obj)

    @classmethod
    def from_data(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def reduce(cls, structs, weights=None, **kwargs):
        raise NotImplementedError
