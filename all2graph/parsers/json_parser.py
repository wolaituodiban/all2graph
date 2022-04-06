import json
from datetime import datetime as ddt
from typing import List, Union, Dict, Set, Tuple

import pandas as pd

from .data_parser import DataParser
from ..graph import RawGraph
from ..utils import tqdm
from ..globals import READOUT, ITEM


class JsonParser(DataParser):
    def __init__(
            self,
            # dataframe 列名
            json_col,
            time_col,
            time_format=None,
            targets: Union[List, Dict] = None,
            # 图生成参数
            dense_dict=False,
            dict_degree=1,
            list_degree=1,
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
            dict_degree: 自然数，插入dict时跳连前置节点的度数
            dense_dict: 字典内是否有边
            list_degree: 自然数，插入list时跳连前置节点的度数
            l_inner_degree: 整数，list内部节点跳连后置节点的度数，负数表示全部
            r_l_inner_degree: 整数，list内部节点跳连前置节点的度数，负数表示全部
            seq_keys: 是否生成网格
            lid_keys: 样本内表示id的key
            gid_keys: 样本间表示id的key
            processor:
                def processor(json_obj, now=None, tokenizer=None, **kwargs):
                        new_json_obj = ...
                        return new_json_obj
            **kwargs:
        """
        super().__init__(data_col=json_col, time_col=time_col, time_format=time_format, targets=targets, **kwargs)
        self.dense_dict = dense_dict
        self.dict_degree = dict_degree
        self.list_degree = list_degree
        self.lid_keys = lid_keys
        self.gid_keys = gid_keys
        self.processor = processor

    def _add_dict(self, graph: RawGraph, obj: dict, vids: List[int]):
        sub_vids = vids[-self.dict_degree:]
        nids = []
        for key, value in obj.items():
            if self.lid_keys and key in self.lid_keys:
                # local id
                nid = graph.add_lid_(key, value)
                graph.add_edge_(sub_vids[-1], nid)
                graph.add_edges_([nid] * len(sub_vids), sub_vids)
            elif self.gid_keys and key in self.gid_keys:
                # global id
                nid = graph.add_gid_(key, value)
                graph.add_edge_(sub_vids[-1], nid)
                graph.add_edges_([nid] * len(sub_vids), sub_vids)
            else:
                nid = graph.add_kv_(key, value)
                graph.add_edges_([nid] * len(sub_vids), sub_vids)
                self.add_obj(graph, obj=value, key=key, vids=vids + [nid])
            nids.append(nid)
        if self.dense_dict:
            graph.add_dense_edges_(nids)

    def _add_list(self, graph: RawGraph, key, obj: list, vids: List[int]):
        sub_vids = vids[-self.dict_degree:]
        for value in obj:
            nid = graph.add_kv_(key, value)
            graph.add_edges_([nid] * len(sub_vids), sub_vids)
            self.add_obj(graph, obj=value, vids=vids+[nid], key=key)

    def add_obj(self, graph, obj, key=READOUT, vids=None):
        vids = vids or [graph.add_kv_(key, obj)]
        if isinstance(obj, dict):
            self._add_dict(graph, obj=obj, vids=vids)
        elif isinstance(obj, list):
            self._add_list(graph, obj=obj, vids=vids, key='_'.join([key, ITEM]))

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
        cols = [self.data_col, self.time_col]
        for row in tqdm(df[cols].itertuples(), disable=disable, postfix='parsing json'):
            obj = self.process_json(row[1], now=row[2])
            graph.add_sample_()
            self.add_obj(graph, obj=obj)
        graph.add_targets_(self.targets)
        return graph


