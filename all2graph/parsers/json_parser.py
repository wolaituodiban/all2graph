import json
from datetime import datetime as ddt
from typing import List, Union, Dict, Set, Tuple

import pandas as pd

from .data_parser import DataParser
from ..graph.raw_graph import RawGraph
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
            local_foreign_key_types: Set[Union[str, Tuple[str]]] = None,
            global_foreign_key_types: Set[Union[str, Tuple[str]]] = None,
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
            dense_dict: 字典内是否有边
            dict_degree: 自然数, 插入dict时跳连前置节点的度数
            list_degree: 自然数, 插入list时跳连前置节点的度数
            local_foreign_key_types: 样本内表示id的key
            global_foreign_key_types: 样本间表示id的key
            processor:
                def processor(json_obj, now=None, **kwargs):
                        new_json_obj = ...
                        return new_json_obj
            **kwargs:
        """
        super().__init__(data_col=json_col, time_col=time_col, time_format=time_format, targets=targets, **kwargs)
        self.dense_dict = dense_dict
        self.dict_degree = dict_degree
        self.list_degree = list_degree
        self.local_foreign_key_types = local_foreign_key_types
        self.global_foreign_key_types = global_foreign_key_types
        self.processor = processor

    def to_json(self) -> dict:
        outputs = super().to_json()
        outputs['json_col'] = outputs['data_col']
        del outputs['data_col']
        outputs['dense_dict'] = self.dense_dict
        outputs['dict_degree'] = self.dict_degree
        outputs['list_degree'] = self.list_degree
        if self.local_foreign_key_types is not None:
            outputs['local_foreign_key_types'] = list(self.local_foreign_key_types)
        if self.global_foreign_key_types is not None:
            outputs['global_foreign_key_types'] = list(self.global_foreign_key_types)
        return outputs

    def _add_dict(self, graph, sample, obj, vids):
        """

        Args:
            graph: RawGraph
            sample: int
            obj: dict
            vids: List[int]

        Returns:

        """
        sub_vids = vids[-self.dict_degree:]
        nids = []
        for key, value in obj.items():
            if self.local_foreign_key_types and key in self.local_foreign_key_types:
                # local_foreign_key
                nid = graph.add_local_foreign_key_(sample, key, value)
                graph.add_edge_(sub_vids[-1], nid)
                graph.add_edges_([nid] * len(sub_vids), sub_vids)
            elif self.global_foreign_key_types and key in self.global_foreign_key_types:
                # global_foreign_key
                nid = graph.add_global_foreign_key_(sample, key, value)
                graph.add_edge_(sub_vids[-1], nid)
                graph.add_edges_([nid] * len(sub_vids), sub_vids)
            else:
                nid = graph.add_kv_(sample, key, value)
                graph.add_edges_([nid] * len(sub_vids), sub_vids)
                self.add_obj(graph, sample, obj=value, key=key, vids=vids + [nid])
            nids.append(nid)
        if self.dense_dict:
            graph.add_dense_edges_(nids)

    def _add_list(self, graph, sample, key, obj, vids):
        """

        Args:
            graph: RawGraph
            sample: int
            key: str
            obj: list
            vids: List[int]

        Returns:

        """
        sub_vids = vids[-self.dict_degree:]
        for value in obj:
            nid = graph.add_kv_(sample, key, value)
            graph.add_edges_([nid] * len(sub_vids), sub_vids)
            self.add_obj(graph, sample, obj=value, vids=vids+[nid], key=key)

    def add_obj(self, graph, sample, obj, key=READOUT, vids=None):
        """

        Args:
            graph: RawGraph
            sample: int
            obj:
            key:
            vids:

        Returns:

        """
        vids = vids or [graph.add_kv_(sample, key, obj)]
        if isinstance(obj, dict):
            self._add_dict(graph, sample, obj=obj, vids=vids)
        elif isinstance(obj, list):
            self._add_list(graph, sample, obj=obj, vids=vids, key='_'.join([key, ITEM]))

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
        for i, row in tqdm(enumerate(df[cols].itertuples()), disable=disable, postfix='parsing json'):
            obj = self.process_json(row[1], now=row[2])
            self.add_obj(graph, i, obj=obj)
        graph.add_targets_(self.targets)
        return graph

    def extra_repr(self) -> str:
        output = [
            super().extra_repr(),
            'dense_dict={}'.format(self.dense_dict),
            'dict_degree={}'.format(self.dict_degree),
            'list_degree={}'.format(self.list_degree),
            'local_foreign_key_types={}'.format(self.local_foreign_key_types),
            'global_foreign_key_types={}'.format(self.global_foreign_key_types),
            'processor={}'.format(self.processor)
        ]
        return '\n'.join(output)
