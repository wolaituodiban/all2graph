from typing import Dict, List, Union, Set, Iterable

import jieba
from toad.utils.progress import Progress
from ..graph import Graph
from .resolver import Resolver


class JsonResolver(Resolver):
    def __init__(
            self,
            flatten_dict=False,
            dict_pred_degree=1,
            list_pred_degree=1,
            list_inner_degree=-1,
            r_list_inner_degree=-1,
            local_index_names: Set[str] = None,
            global_index_names: Set[str] = None,
            segmentation=False
    ):
        """

        :param flatten_dict:
        :param dict_pred_degree:
        :param list_pred_degree:
        :param list_inner_degree:
        :param r_list_inner_degree:
        :param local_index_names:
        """
        self.flatten_dict = flatten_dict
        self.dict_pred_degree = dict_pred_degree
        self.list_pred_degree = list_pred_degree
        self.list_inner_degree = list_inner_degree
        self.r_list_inner_degree = r_list_inner_degree
        self.local_index_names = local_index_names
        self.global_index_names = global_index_names
        self.segmentation = segmentation

    def insert_component(
            self,
            graph: Graph,
            component_id: int,
            name: str,
            value: Union[Dict, List, str, int, float, None],
            preds: Union[List[int], None],
            local_index_mapper: Dict[str, int],
            global_index_mapper: Dict[str, int]
    ):
        """
        插入一个连通片（component）。如果图中任意两点都是连通的，那么图被称作连通图。
        :param graph:
        :param component_id: 连通片编号
        :param name: 第一个节点的名称
        :param value: 第一个节点的值
        :param preds: 前置节点的编号
        :param local_index_mapper: index的value和node_id的映射
        :param global_index_mapper: index的value和node_id的映射
        :return:
        """
        if preds is None:
            node_id = graph.insert_node(component_id, name, value)
            self.insert_component(
                graph=graph, component_id=component_id, name=name, value=value, preds=[node_id],
                local_index_mapper=local_index_mapper, global_index_mapper=global_index_mapper
            )
        elif isinstance(value, dict):
            for k, v in value.items():
                if self.local_index_names is not None and k in self.local_index_names:
                    if v in local_index_mapper:
                        node_id = local_index_mapper[v]
                    else:
                        node_id = graph.insert_node(component_id, k, v)
                        local_index_mapper[v] = node_id
                    new_preds = preds
                    new_succs = [node_id] * len(preds)
                    graph.insert_edges(new_preds + new_succs, new_succs + new_preds)
                elif self.global_index_names is not None and k in self.global_index_names:
                    if v in global_index_mapper:
                        node_id = global_index_mapper[v]
                    else:
                        node_id = graph.insert_node(component_id, k, v)
                        global_index_mapper[v] = node_id
                    new_preds = preds
                    new_succs = [node_id] * len(preds)
                    graph.insert_edges(new_preds + new_succs, new_succs + new_preds)
                elif self.flatten_dict and isinstance(v, dict):
                    self.insert_component(
                        graph=graph, component_id=component_id, name=k, value=v, preds=preds,
                        local_index_mapper=local_index_mapper, global_index_mapper=global_index_mapper
                    )
                else:
                    node_id = graph.insert_node(component_id, k, v)
                    new_preds = preds[-self.dict_pred_degree:]
                    new_succs = [node_id] * len(new_preds)
                    graph.insert_edges(new_preds, new_succs)
                    self.insert_component(
                        graph=graph, component_id=component_id, name=k, value=v, preds=preds + [node_id],
                        local_index_mapper=local_index_mapper, global_index_mapper=global_index_mapper
                    )
        elif isinstance(value, list) or (self.segmentation and isinstance(value, str)):
            node_ids = []
            if isinstance(value, str):
                temp_value = [v.lower() for v in jieba.cut(value)]
                # 修改之前插入的value
                if len(temp_value) == 1:
                    graph.values[preds[-1]] = temp_value[0]
                    return
                else:
                    graph.values[preds[-1]] = []
            else:
                temp_value = value
            for v in temp_value:
                node_id = graph.insert_node(component_id, name, v)

                new_preds = preds[-self.list_pred_degree:]
                if self.list_inner_degree >= 0:
                    new_preds += node_ids[-self.list_inner_degree:]

                new_succs = [node_id] * len(new_preds)
                if self.r_list_inner_degree >= 0:
                    new_succs += node_ids[-self.r_list_inner_degree:]
                    new_preds += [node_id] * (len(new_succs) - len(new_preds))

                graph.insert_edges(new_preds, new_succs)
                node_ids.append(node_id)
                if isinstance(value, list):
                    self.insert_component(
                        graph=graph, component_id=component_id, name=name, value=v, preds=preds + [node_id],
                        local_index_mapper=local_index_mapper, global_index_mapper=global_index_mapper
                    )

    def resolve(
            self,
            root_name: str,
            jsons: Iterable[Union[Dict, List]],
            progress_bar=False,
    ) -> (Graph, dict, List[dict]):
        graph = Graph()
        global_index_mapper = {}
        local_index_mappers = []
        process = jsons
        if progress_bar:
            process = Progress(jsons)
            process.suffix = 'resolving json'
        for i, value in enumerate(process):
            local_index_mapper = {}
            self.insert_component(
                graph=graph, component_id=i, name=root_name, value=value, preds=None,
                local_index_mapper=local_index_mapper, global_index_mapper=global_index_mapper
            )
            local_index_mappers.append(local_index_mapper)
        return graph, global_index_mapper, local_index_mappers
