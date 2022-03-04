from typing import List, Union, Tuple, Set, Dict

import numpy as np
import pandas as pd

from ..info import MetaInfo, GraphInfo
from ..meta_struct import MetaStruct
from ..utils import tqdm
from ..globals import *


def gen_edges_for_seq_(nids, degree, r_degree):
    u, v = [], []
    for i, nid in enumerate(nids):
        # 正向
        end = i + degree + 1 if degree >= 0 else len(nids)
        v += nids[i + 1:end]
        # 反向
        start = max(0, i - r_degree) if r_degree >= 0 else 0
        v += nids[start:i]
        # 补全
        u += [nid] * (len(v) - len(u))
    return u, v


class RawGraph(MetaStruct):
    """
    异构图
    有三种类型的点：
        key：代表entity或者readout的类型，拥有一个属性
            value：表示key本身的值
        value：代表实体，拥用两个属性
            sample id：表示样本编号
            value：表示entity本身的值
        readout：代表目标
    有五种类型的边：
        key2key：多对多
        key2value：一对一
        key2readout：一对一
        value2value：多对多
        value2readout：一对多
    有一个graph level的属性：
        readout：记录所有是readout的value的id，所有与readout相连的value都是root，但是readout不一定与root相连
                 每一个sample的root必须是唯一的
    """
    def __init__(self, **kwargs):
        super().__init__(initialized=True, **kwargs)
        self.keys = []
        self.values = []
        self.edges = {
            KEY2KEY: [[], []],
            KEY2VALUE: [[], []],
            VALUE2VALUE: [[], []],
            SAMPLE2VALUE: [[], []],
        }
        # 每个sample的第一个node自动被作为root
        self.__roots: List[int] = []

        # 用于加快__add_k_的速度
        self.__ori_kids: Dict[Union[str, Tuple[str]], int] = {}  # {key: kid}

        # 记录作为要被当作id的value的id
        self.__lids: Dict[int, Dict[str, int]] = {}  # {sid: {value: vid}}
        self.__gids: Dict[str, int] = {}  # {value: vid}

    def _assert(self):
        # 检验sample的数量一致
        all_sids = []
        for etype, (u, v) in self.edges.items():
            assert len(u) == len(v)
            if etype[0] == SAMPLE:
                all_sids += u
            if etype[-1] == SAMPLE:
                all_sids += v
        all_sids = set(all_sids)
        assert len(self.__roots) == len(all_sids), (self.__roots, all_sids)
        if len(all_sids) > 0:
            assert len(all_sids) == max(all_sids) + 1
        # 检验key的数量一致
        all_kids = []
        for etype, (u, v) in self.edges.items():
            if etype[0] == KEY:
                all_kids += u
            if etype[-1] == KEY:
                all_kids += v
        all_kids = set(all_kids)
        assert len(self.keys) == len(all_kids) == max(all_kids) + 1, (self.keys, all_kids)
        # 检验value的数量一致
        all_vids = []
        for etype, (u, v) in self.edges.items():
            if etype[0] == VALUE:
                all_vids += u
            if etype[-1] == VALUE:
                all_vids += v
        all_vids = set(all_vids)
        assert len(self.values) == len(all_vids) == max(all_vids) + 1
        # 检验顺序
        for i in range(1, len(self.values)):
            assert self.edges[KEY2VALUE][1][i] - self.edges[KEY2VALUE][1][i - 1] == 1

    @property
    def num_samples(self):
        return len(self.__roots)

    @property
    def formated_values(self):
        # 排除所有ID的value
        glids = set(self.__gids.values())
        for lids in self.__lids.values():
            glids = glids.union(lids.values())
        return [None if isinstance(v, (list, dict)) or i in glids else v for i, v in enumerate(self.values)]

    @property
    def num_values(self):
        return self.num_nodes(VALUE)

    @property
    def ntypes(self) -> Set[str]:
        ntypes = []
        for utype, _, vtype in self.edges:
            ntypes.append(utype)
            ntypes.append(vtype)
        return set(ntypes)

    @property
    def readout_types(self):
        return {ntype for ntype in self.ntypes if ntype != KEY and ntype != VALUE and ntype != SAMPLE}

    def __eq__(self, other):
        return super().__eq__(other) \
               and self.keys == other.keys \
               and self.values == other.values \
               and self.edges == other.edges \
               and self.__roots == other.__roots \
               and self.__ori_kids == other.__ori_kids \
               and self.__lids == other.__lids \
               and self.__gids == other.__gids

    def num_nodes(self, ntype=VALUE):
        if ntype == KEY:
            return len(self.keys)
        elif ntype == SAMPLE:
            return self.num_samples
        elif ntype == VALUE:
            return len(self.values)
        elif ntype is None:
            return sum(map(self.num_nodes, self.ntypes))
        else:
            return len(self.edges[SAMPLE, EDGE, ntype][0])

    def num_edges(self, etype=VALUE2VALUE):
        if etype is None:
            return sum(map(self.num_edges, self.edges))
        else:
            return len(self.edges[etype][0])

    def get_keys(self, nids, ntype=VALUE):
        if ntype == KEY:
            return [self.keys[i] for i in nids]
        else:
            ntype2key = {v: u for u, v in zip(*self.edges[KEY, EDGE, ntype])}
            return [self.keys[ntype2key[i]] for i in nids]

    def get_values(self, nids, ntype=VALUE):
        if ntype != VALUE:
            return [None] * len(nids)
        return [self.values[i] for i in nids]

    def get_sids(self, nids, ntype=VALUE):
        etype = (SAMPLE, EDGE, ntype)
        ntype2sample = {v: u for u, v in zip(*self.edges[etype])}
        return [ntype2sample[i] for i in nids]

    def add_edge_(self, u, v, etype=VALUE2VALUE, bidirectional=False):
        self.edges[etype][0].append(u)
        self.edges[etype][1].append(v)
        if bidirectional:
            self.edges[etype][0].append(v)
            self.edges[etype][1].append(u)

    def add_edges_(self, u: List[int], v: List[int], etype=VALUE2VALUE, bidirectional=False):
        if bidirectional:
            u, v = u + v, v + u
        if etype not in self.edges:
            self.edges[etype] = [[], []]
        self.edges[etype][0] += u
        self.edges[etype][1] += v

    # todo 考虑一些操作是否在Graph处做比较快
    def add_edges_for_seq_(self, nids: List[int], etype=VALUE2VALUE, degree: int = -1, r_degree: int = -1):
        """
        为一些列点之间添加边
        Args:
            nids: 点坐标
            etype: 边类型
            degree: 正向度数，-1表示全部，0表示无
            r_degree: 反向度数，-1表示全部，0表示无

        Returns:

        """
        u, v = gen_edges_for_seq_(sorted(nids), degree=degree, r_degree=r_degree)
        self.add_edges_(u, v, etype=etype)

    def add_edges_by_key_(self, keys=None, **kwargs):
        """
        为某一个key的所有value点组成点序列增加边
        Args:
            keys:
            **kwargs: add_edges_for_seq的参数

        Returns:

        """
        group = {}
        value2sample = {v: s for s, v in zip(*self.edges[SAMPLE2VALUE])}
        for u, v in zip(*self.edges[KEY2VALUE]):
            k = (value2sample[v], u)
            if k not in group:
                group[k] = []
            group[k].append(v)

        for (sid, u), v in group.items():
            if keys and self.keys[u] not in keys:
                continue
            self.add_edges_for_seq_(v, **kwargs)

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
            kid = self.num_nodes(KEY)
            self.__ori_kids[key] = kid
            self.keys.append(key)
            if isinstance(key, tuple):
                self.keys += list(key)
                kids = list(range(self.__ori_kids[key], len(self.keys)))
                self.add_edges_for_seq_(kids, KEY2KEY)
        return self.__ori_kids[key]

    def __add_v_(self, sid, value) -> int:
        """
        # 自动添加root机制
        # 每一个sid的第一个点会被作为root
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
        self.values.append(value)
        self.add_edge_(sid, vid, etype=SAMPLE2VALUE)
        if sid >= len(self.__roots):
            self.__roots.append(vid)
            # 与所有global id节点相连
            u = [sid] * len(self.__gids)
            v = list(self.__gids.values())
            self.add_edges_(u, v, etype=SAMPLE2VALUE)
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
            flag = True
            vid = self.__add_v_(0, value)
            self.__gids[value] = vid
            # 与所有sample节点相连
            u = list(range(1, len(self.__roots)))
            v = [vid] * len(u)
            self.add_edges_(u, v, etype=SAMPLE2VALUE)
        return self.__gids[value], flag

    def add_kv_(self, sid: int, key: Union[str, Tuple], value) -> int:
        """返回新增的entity的id"""
        kid = self.__add_k_(key)
        vid = self.__add_v_(sid, value)
        self.add_edge_(kid, vid, KEY2VALUE)
        return vid

    def add_readouts_(self, ntypes: Union[List[str], Dict[str, Union[str, Tuple[str]]]]):
        assert self.ntypes.isdisjoint(ntypes), '{} already exists'.format(self.ntypes.intersection(ntypes))
        if isinstance(ntypes, list):
            ntypes = {_: _ for _ in ntypes}
        for ori_key, mapped_key in ntypes.items():
            assert isinstance(ori_key, str)
            kid = self.__add_k_(mapped_key)
            tids = list(range(len(self.__roots)))
            # key to readout
            self.add_edges_([kid] * len(tids), tids, (KEY, EDGE, ori_key))
            # value to readout
            self.add_edges_(self.__roots, tids, (VALUE, EDGE, ori_key))
            # sample to readout
            self.add_edges_(tids, tids, (SAMPLE, EDGE, ori_key))

    def add_lid_(self, sid: int, key: Union[str, Tuple[str]], value: str) -> int:
        """如果id已存在，则不会对图产生任何修改"""
        kid = self.__add_k_(key)
        vid, flag = self.__add_lv_(sid, value)
        if flag:
            self.add_edge_(kid, vid, KEY2VALUE)
        return vid

    def add_gid_(self, key: Union[str, Tuple[str]], value: str) -> int:
        """如果id已存在，则不会对图产生任何修改"""
        kid = self.__add_k_(key)
        vid, flag = self.__add_gv_(value)
        if flag:
            self.add_edge_(kid, vid, KEY2VALUE)
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

        # value & readout
        for ntype in self.ntypes:
            if ntype == KEY or ntype == SAMPLE:
                continue
            u, v = self.edges[(VALUE, EDGE, ntype)]
            sids += self.get_sids(u)
            uids += u
            vids += v
            utypes += [VALUE] * len(u)
            vtypes += [ntype] * len(v)
            u_keys += self.get_keys(u, VALUE)
            v_keys += self.get_keys(v, ntype)
            u_values += self.get_values(u)
            v_values += self.get_values(v)

        if key:
            # key graph
            for ntype in self.ntypes:
                if ntype == SAMPLE:
                    continue
                u, v = self.edges[(KEY, EDGE, ntype)]
                sids += [None] * len(u)
                uids += u
                vids += v
                utypes += [KEY] * len(u)
                vtypes += [ntype] * len(v)
                u_keys += [KEY] * len(u)
                u_values += self.get_keys(u, KEY)
                if ntype == KEY:
                    v_keys += [KEY] * len(v)
                    v_values += self.get_keys(v, ntype)
                else:
                    v_keys += self.get_keys(v, ntype)
                    v_values += self.get_values(v, ntype)

        df = pd.DataFrame({SAMPLE: sids, 'u': uids,  'utype': utypes, 'u_key': u_keys, 'u_value': u_values,
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
            graph.add_node(u, **{SAMPLE: row[SAMPLE], KEY: row.u_key, VALUE: row.u_value})
            graph.add_node(v, **{SAMPLE: row[SAMPLE], KEY: row.v_key, VALUE: row.v_value})
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

    def meta_info(self, **kwargs) -> MetaInfo:
        vids = list(range(self.num_nodes(VALUE)))
        sample_ids = self.get_sids(vids)
        keys = self.get_keys(vids)
        values = self.formated_values
        for ntype in self.readout_types:
            u, v = self.edges[(SAMPLE, EDGE, ntype)]
            sample_ids += u
            keys += self.get_keys(v, ntype=ntype)
            values += [None] * len(v)
        return GraphInfo.from_data(sample_ids=sample_ids, keys=keys, values=values, **kwargs)

    def extra_repr(self) -> str:
        return 'num_nodes={},\nnum_edges={}'.format(
            {ntype: self.num_nodes(ntype) for ntype in self.ntypes},
            {etype: self.num_edges(etype) for etype in self.edges}
        )
