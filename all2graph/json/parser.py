import json
from inspect import ismethod
from datetime import datetime as ddt
from typing import Dict, List, Union, Set

import pandas as pd


from jsonpromax import JsonPathTree
from ..parsers import DataParser
from ..graph import RawGraph
from ..utils import tqdm, Tokenizer, default_tokenizer


class JsonParser(DataParser):
    def __init__(
            self,
            # dataframe 列名
            json_col,
            time_col,
            time_format=None,
            # 图生成参数
            flatten_dict=False,
            dict_dst_degree=1,
            list_dst_degree=1,
            list_inner_degree=1,
            r_list_inner_degree=-1,
            local_id_keys: Set[str] = None,
            global_id_keys: Set[str] = None,
            segment_value=False,
            self_loop=True,
            # 预处理
            processor=None,
            processors=None,
            tokenizer: Tokenizer = None,
            error=True,
            warning=True,
            **kwargs
    ):
        """

        Args:
            json_col:
            time_col:
            time_format:
            flatten_dict:
            dict_dst_degree: 自然数，插入dict时跳连前置节点的度数，0表示全部
            list_dst_degree: 自然数，插入list时跳连前置节点的度数，0表示全部
            list_inner_degree: 整数，list内部节点跳连前置节点的度数，0表述全部，-1表示没有
            r_list_inner_degree: 整数，list内部节点跳连后置节点的度数，0表述全部，-1表示没有
            local_id_keys:
            global_id_keys:
            segment_value:
            self_loop:
            processor: callable
                    def processor(json_obj, now=None, tokenizer=None, **kwargs):
                        new_json_obj = ...
                        return new_json_obj
            processors: JsonPathTree的参数,
            tokenizer: 默认使用None
            error: 如果遇到错误，会报错
            warning: 如果遇到错误，会报警
            **kwargs:
        """
        super().__init__(json_col=json_col, time_col=time_col, time_format=time_format, **kwargs)
        self.flatten_dict = flatten_dict
        self.dict_dst_degree = dict_dst_degree
        self.list_dst_degree = list_dst_degree
        self.list_inner_degree = list_inner_degree
        self.r_list_inner_degree = r_list_inner_degree
        self.local_id_keys = local_id_keys
        self.global_id_keys = global_id_keys
        self.segment_value = segment_value
        self.self_loop = self_loop
        if processor is not None:
            self.json_path_tree = processor
        elif processors is not None:
            print('processors is depreciated, please use procesor')
            self.json_path_tree = JsonPathTree(processors=processors)
        else:
            self.json_path_tree = None
        self.tokenizer = tokenizer
        if self.segment_value and self.tokenizer is None:
            self.tokenizer = default_tokenizer()
        self.error = error
        self.warning = warning

        self._enable_preprocessing = True

    def enable_preprocessing(self):
        self._enable_preprocessing = True

    def disable_preprocessing(self):
        self._enable_preprocessing = False

    def insert_dict(
            self,
            graph: RawGraph,
            component_id: int,
            value: Union[Dict, List, str, int, float, None],
            dsts: Union[List[int], None],
            local_index_mapper: Dict[str, int],
            global_index_mapper: Dict[str, int],
    ):
        # local index和global index的逻辑有点问题
        for k, v in value.items():
            if self.local_id_keys is not None and k in self.local_id_keys:
                if v in local_index_mapper:
                    node_id = local_index_mapper[v]
                else:
                    node_id = graph.insert_node(component_id, k, v, self_loop=self.self_loop)
                    local_index_mapper[v] = node_id
                graph.insert_edges(dsts=[dsts[-1]], srcs=[node_id], bidirection=True)
            elif self.global_id_keys is not None and k in self.global_id_keys:
                if v in global_index_mapper:
                    node_id = global_index_mapper[v]
                else:
                    node_id = graph.insert_node(component_id, k, v, self_loop=self.self_loop)
                    global_index_mapper[v] = node_id
                graph.insert_edges(dsts=[dsts[-1]], srcs=[node_id], bidirection=True)
            elif self.flatten_dict and isinstance(v, dict):
                self.insert_component(
                    graph=graph, component_id=component_id, key=k, value=v, dsts=dsts,
                    local_index_mapper=local_index_mapper, global_index_mapper=global_index_mapper)
            else:
                node_id = graph.insert_node(component_id, k, v, self_loop=self.self_loop)
                new_dsts = dsts[-self.dict_dst_degree:]
                new_srcs = [node_id] * len(new_dsts)
                graph.insert_edges(dsts=new_dsts, srcs=new_srcs)
                self.insert_component(
                    graph=graph, component_id=component_id, key=k, value=v, dsts=dsts + [node_id],
                    local_index_mapper=local_index_mapper, global_index_mapper=global_index_mapper)

    def insert_array(
            self,
            graph: RawGraph,
            component_id: int,
            key: str,
            value: Union[Dict, List, str, int, float, None],
            dsts: Union[List[int], None],
            local_index_mapper: Dict[str, int],
            global_index_mapper: Dict[str, int],
    ):
        recursive_flag = True
        if isinstance(value, str):
            recursive_flag = False
            value = self.tokenizer.lcut(value)
            if len(value) < 2:
                return
            # 修改之前插入的value
            graph.value[dsts[-1]] = value

        node_ids = []
        for v in value:
            node_id = graph.insert_node(component_id, key, v, self_loop=self.self_loop)

            new_dsts = dsts[-self.list_dst_degree:]
            new_srcs = [node_id] * len(new_dsts)
            if self.list_inner_degree >= 0:
                new_srcs += node_ids[-self.list_inner_degree:]
                new_dsts += [node_id] * (len(new_srcs) - len(new_dsts))

            if self.r_list_inner_degree >= 0:
                new_dsts += node_ids[-self.r_list_inner_degree:]
                new_srcs += [node_id] * (len(new_dsts) - len(new_srcs))

            graph.insert_edges(dsts=new_dsts, srcs=new_srcs)
            node_ids.append(node_id)
            if recursive_flag:
                self.insert_component(
                    graph=graph, component_id=component_id, key=key, value=v, dsts=dsts + [node_id],
                    local_index_mapper=local_index_mapper, global_index_mapper=global_index_mapper)

    def insert_component(
            self,
            graph: RawGraph,
            component_id: int,
            value: Union[Dict, List, str, int, float, None],
            dsts: List[int],
            local_index_mapper: Dict[str, int],
            global_index_mapper: Dict[str, int],
            key: str = None,
    ):
        """
        插入一个连通片（component）。如果图中任意两点都是连通的，那么图被称作连通图。
        :param graph:
        :param component_id: 连通片编号
        :param key: 第一个节点的名称
        :param value: 第一个节点的值
        :param dsts: 前置节点的编号
        :param local_index_mapper: index的value和node_id的映射
        :param global_index_mapper: index的value和node_id的映射
        :return:
        """
        if key is None:
            readout_id = graph.insert_readout(component_id, value=value, self_loop=self.self_loop)
            self.insert_component(
                graph=graph, component_id=component_id, key=graph.key[-1], value=value, dsts=[readout_id],
                local_index_mapper=local_index_mapper, global_index_mapper=global_index_mapper)
        elif isinstance(value, dict):
            self.insert_dict(
                graph=graph, component_id=component_id, value=value, dsts=dsts,
                local_index_mapper=local_index_mapper, global_index_mapper=global_index_mapper)
        elif isinstance(value, list) or (self.segment_value and isinstance(value, str)):
            self.insert_array(
                graph=graph, component_id=component_id, key=key, value=value, dsts=dsts,
                local_index_mapper=local_index_mapper, global_index_mapper=global_index_mapper)

    def parse_json(self, obj, now=None):
        # json load
        try:
            obj = json.loads(obj)
        except (json.JSONDecodeError, TypeError, ValueError, KeyError, IndexError):
            pass

        # json预处理
        if self.json_path_tree is not None:
            if pd.notna(now):
                now = ddt.strptime(now, self.time_format)
            obj = self.json_path_tree(obj, now=now, tokenizer=self.tokenizer)
        return obj

    def parse_df(
            self,
            df: pd.DataFrame,
            disable: bool = True,
    ):
        for obj, now in tqdm(zip(df[self.json_col], df[self.time_col]), disable=disable, postfix='parsing json'):
            yield self.parse_json(obj, now)

    def save(self, df, dst, disable=True):
        assert self.global_id_keys is None
        self.enable_preprocessing()

        local_index_mappers = []
        graphs = []
        for obj in self.parse_df(df=df, disable=disable):
            graph = RawGraph()
            local_index_mapper = {}
            self.insert_component(
                graph=graph, component_id=0, value=obj, dsts=[],
                local_index_mapper=local_index_mapper, global_index_mapper={})
            graphs.append(graph)
            local_index_mappers.append(local_index_mapper)

        df = df.copy()
        df[self.json_col] = [json.dumps(graph.to_json(drop_nested_value=True)) for graph in graphs]
        df['local_index_mapper'] = list(map(json.dumps, local_index_mappers))
        df.to_csv(dst)

    def parse(
            self,
            df: pd.DataFrame,
            disable: bool = True,
    ) -> (RawGraph, dict, List[dict]):
        global_index_mapper = {}
        if self._enable_preprocessing:
            graph = RawGraph()
            local_index_mappers = []
            for component_id, obj in enumerate(self.parse_df(df=df, disable=disable)):
                local_index_mapper = {}
                self.insert_component(
                    graph=graph, component_id=component_id, value=obj, dsts=[],
                    local_index_mapper=local_index_mapper, global_index_mapper=global_index_mapper)
                local_index_mappers.append(local_index_mapper)
        else:
            assert self.global_id_keys is None
            graph = RawGraph.batch(RawGraph.from_json(json.loads(obj)) for obj in df[self.json_col])
            local_index_mappers = [json.loads(obj) for obj in df['local_index_mapper']]
        return graph, global_index_mapper, local_index_mappers

    def extra_repr(self) -> str:
        s = ', '.join(
            '{}={}'.format(k, v) for k, v in self.__dict__.items()
            if not ismethod(v) and not isinstance(v, JsonPathTree) and not k.startswith('_')
        )
        if self.json_path_tree is not None:
            s += '\njson_path_tree:'
            for line in str(self.json_path_tree).split('\n'):
                s += '\n\t' + line
        return s
