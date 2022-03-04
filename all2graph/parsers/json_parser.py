import json
from datetime import datetime as ddt
from typing import List, Union, Dict, Set, Tuple

import pandas as pd

from .data_parser import DataParser
from ..graph import RawGraph
from ..utils import tqdm
from ..globals import ROOT, ITEM


class JsonParser(DataParser):
    def __init__(
            self,
            # dataframe 列名
            json_col,
            time_col,
            time_format=None,
            targets: Union[List, Dict] = None,
            # 图生成参数
            d_degree=1,
            d_inner_edge=False,
            l_degree=1,
            to_bidirected=False,
            lid_keys: Set[Union[str, Tuple[str]]] = None,
            gid_keys: Set[Union[str, Tuple[str]]] = None,
            # 预处理
            processor=None,
            **kwargs
    ):
        """

        Args:
            json_col:
            time_col:
            time_format:
            targets:
            d_degree: 自然数，插入dict时跳连前置节点的度数
            d_inner_edge: 字典内是否有边
            l_degree: 自然数，插入list时跳连前置节点的度数
            l_inner_degree: 整数，list内部节点跳连后置节点的度数，负数表示全部
            r_l_inner_degree: 整数，list内部节点跳连前置节点的度数，负数表示全部
            to_bidirected: 双向边
            seq_keys: 是否生成网格
            lid_keys: 样本内表示id的key
            gid_keys: 样本间表示id的key
            processor:
                def processor(json_obj, now=None, tokenizer=None, **kwargs):
                        new_json_obj = ...
                        return new_json_obj
            **kwargs:
        """
        super().__init__(json_col=json_col, time_col=time_col, time_format=time_format, targets=targets, **kwargs)
        self.d_degree = d_degree
        self.d_inner_edge = d_inner_edge
        self.l_degree = l_degree
        self.bidirectional = to_bidirected
        self.lid_keys = lid_keys
        self.gid_keys = gid_keys
        self.processor = processor

    def _add_dict(self, graph: RawGraph, sid: int, obj: dict, vids: List[int]):
        sub_vids = vids[-self.d_degree:]
        nids = []
        for key, value in obj.items():
            if self.lid_keys and key in self.lid_keys:
                # local id
                nid = graph.add_lid_(sid, key, value)
                graph.add_edges_([nid] * len(sub_vids), sub_vids, bidirectional=True)
            elif self.gid_keys and key in self.gid_keys:
                # global id
                nid = graph.add_gid_(key, value)
                graph.add_edges_([nid] * len(sub_vids), sub_vids, bidirectional=True)
            else:
                nid = graph.add_kv_(sid, key, value)
                graph.add_edges_([nid] * len(sub_vids), sub_vids, bidirectional=self.bidirectional)
                self.add_obj(graph, sid=sid, obj=value, key=key, vids=vids + [nid])
            nids.append(nid)
        if self.d_inner_edge:
            graph.add_edges_for_seq_(nids)

    def _add_list(self, graph: RawGraph, sid: int, key, obj: list, vids: List[int]):
        sub_vids = vids[-self.d_degree:]
        for value in obj:
            nid = graph.add_kv_(sid, key, value)
            graph.add_edges_([nid] * len(sub_vids), sub_vids, bidirectional=self.bidirectional)
            self.add_obj(graph, sid=sid, obj=value, vids=vids+[nid], key=key)

    def add_obj(self, graph, sid, obj, key=ROOT, vids=None):
        vids = vids or [graph.add_kv_(sid, key, obj)]
        if isinstance(obj, dict):
            self._add_dict(graph, sid=sid, obj=obj, vids=vids)
        elif isinstance(obj, list):
            self._add_list(
                graph, sid=sid, obj=obj, vids=vids,
                key=key + (ITEM, ) if isinstance(key, tuple) else (key, ITEM))

    def process_json(self, obj, now=None):
        # json load
        if not isinstance(obj, (list, dict)):
            obj = json.loads(obj)

        # json预处理
        if self.processor is not None:
            if pd.notna(now):
                now = ddt.strptime(now, self.time_format)
            obj = self.processor(obj, now=now)
        return obj

    def __call__(self, df: pd.DataFrame, disable: bool = True) -> RawGraph:
        graph = RawGraph()
        cols = [self.json_col, self.time_col]
        for sid, row in tqdm(enumerate(df[cols].itertuples()), disable=disable, postfix='parsing json'):
            obj = self.process_json(row[1], now=row[2])
            self.add_obj(graph, sid=sid, obj=obj)
        self._post_call(graph)
        return graph


