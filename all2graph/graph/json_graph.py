from .graph import Graph
import numpy as np
from typing import Dict, List, Union, Set


class JsonGraph(Graph):
    def __init__(self, flatten_dict=False, dict_pred_degree=1, list_pred_degree=1, list_inner_degree=-1,
                 r_list_inner_degree=-1):
        super().__init__()
        self.flatten_dict = flatten_dict
        self.dict_pred_degree = dict_pred_degree
        self.list_pred_degree = list_pred_degree
        self.list_inner_degree = list_inner_degree
        self.r_list_inner_degree = r_list_inner_degree

    def insert_component(
            self,
            component_id: int,
            name: str,
            value: Union[Dict, List, str, int, float, None],
            preds: List[int] = None,
            index_names: Set[str] = None,
            index_mapper: Dict[str, int] = None
    ):
        """
        插入一个连通片（component）。如果图中任意两点都是连通的，那么图被称作连通图。
        :param component_id: 连通片编号
        :param name: 第一个节点的名称
        :param value: 第一个节点的值
        :param preds: 前置节点的编号
        :param index_names: 这些names会被当作index
        :param index_mapper: index的value和node_id的映射
        :return:
        """
        if preds is None:
            node_id = self.insert_node(component_id, name, value)
            self.insert_component(component_id, name, value, [node_id], index_names, index_mapper)
        elif isinstance(value, dict):
            for k, v in value.items():
                if index_names is not None and k in index_names:
                    if v in index_mapper:
                        node_id = index_mapper[v]
                    else:
                        node_id = self.insert_node(component_id, k, v)
                        index_mapper[v] = node_id
                    new_preds = preds
                    new_succs = [node_id] * len(preds)
                    self.insert_edges(new_preds + new_succs, new_succs + new_preds)
                elif self.flatten_dict and isinstance(v, dict):
                    self.insert_component(component_id, k, v, preds, index_names, index_mapper)
                else:
                    node_id = self.insert_node(component_id, k, v)
                    new_preds = preds[-self.dict_pred_degree:]
                    new_succs = [node_id] * len(new_preds)
                    self.insert_edges(new_preds, new_succs)
                    self.insert_component(component_id, k, v, preds + [node_id], index_names, index_mapper)
        elif isinstance(value, list):
            node_ids = []
            for v in value:
                node_id = self.insert_node(component_id, name, v)

                new_preds = preds[-self.list_pred_degree:]
                if self.list_inner_degree >= 0:
                    new_preds += node_ids[-self.list_inner_degree:]

                new_succs = [node_id] * len(new_preds)
                if self.r_list_inner_degree >= 0:
                    new_succs += node_ids[-self.r_list_inner_degree:]
                    new_preds += [node_id] * (len(new_succs) - len(new_preds))

                self.insert_edges(new_preds, new_succs)
                self.insert_component(component_id, name, v, preds + [node_id], index_names, index_mapper)
                node_ids.append(node_id)
