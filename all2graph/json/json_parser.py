from typing import Dict, List, Union, Set, Iterable

try:
    import jieba
except ImportError:
    jieba = None


from ..data import DataParser
from ..globals import READOUT
from ..graph import Graph
from ..utils import progress_wrapper


class JsonParser(DataParser):
    def __init__(
            self,
            flatten_dict=False,
            dict_pred_degree=1,
            list_pred_degree=1,
            list_inner_degree=-1,
            r_list_inner_degree=-1,
            local_index_names: Set[str] = None,
            global_index_names: Set[str] = None,
            segmentation=False,
            self_loop=False,
    ):
        """

        :param flatten_dict:
        :param dict_pred_degree: 自然数，插入dict时跳连前置节点的度数，0表示全部
        :param list_pred_degree: 自然数，插入list时跳连前置节点的度数，0表示全部
        :param list_inner_degree: 整数，list内部节点跳连前置节点的度数，0表述全部，-1表示没有
        :param r_list_inner_degree: 整数，list内部节点跳连后置节点的度数，0表述全部，-1表示没有
        :param local_index_names:
        :param global_index_names:
        :param segmentation:
        :param self_loop:
        """
        super().__init__()
        self.flatten_dict = flatten_dict
        self.dict_pred_degree = dict_pred_degree
        self.list_pred_degree = list_pred_degree
        self.list_inner_degree = list_inner_degree
        self.r_list_inner_degree = r_list_inner_degree
        self.local_index_names = local_index_names
        self.global_index_names = global_index_names
        self.segmentation = segmentation
        self.self_loop = self_loop

    def insert_dict(
            self,
            graph: Graph,
            component_id: int,
            value: Union[Dict, List, str, int, float, None],
            preds: Union[List[int], None],
            local_index_mapper: Dict[str, int],
            global_index_mapper: Dict[str, int],
            readout_id
    ):
        for k, v in value.items():
            if self.local_index_names is not None and k in self.local_index_names:
                if v in local_index_mapper:
                    node_id = local_index_mapper[v]
                else:
                    node_id = graph.insert_node(component_id, k, v, self_loop=self.self_loop, readout_id=readout_id)
                    local_index_mapper[v] = node_id
                new_preds = preds
                new_succs = [node_id] * len(preds)
                graph.insert_edges(new_preds + new_succs, new_succs + new_preds)
            elif self.global_index_names is not None and k in self.global_index_names:
                if v in global_index_mapper:
                    node_id = global_index_mapper[v]
                else:
                    node_id = graph.insert_node(component_id, k, v, self_loop=self.self_loop, readout_id=readout_id)
                    global_index_mapper[v] = node_id
                new_preds = preds
                new_succs = [node_id] * len(preds)
                graph.insert_edges(new_preds + new_succs, new_succs + new_preds)
            elif self.flatten_dict and isinstance(v, dict):
                self.insert_component(
                    graph=graph, component_id=component_id, name=k, value=v, preds=preds,
                    local_index_mapper=local_index_mapper, global_index_mapper=global_index_mapper,
                    readout_id=readout_id
                )
            else:
                node_id = graph.insert_node(component_id, k, v, self_loop=self.self_loop, readout_id=readout_id)
                new_preds = preds[-self.dict_pred_degree:]
                new_succs = [node_id] * len(new_preds)
                graph.insert_edges(new_preds, new_succs)
                self.insert_component(
                    graph=graph, component_id=component_id, name=k, value=v, preds=preds + [node_id],
                    local_index_mapper=local_index_mapper, global_index_mapper=global_index_mapper,
                    readout_id=readout_id
                )

    def insert_array(
            self,
            graph: Graph,
            component_id: int,
            name: str,
            value: Union[Dict, List, str, int, float, None],
            preds: Union[List[int], None],
            local_index_mapper: Dict[str, int],
            global_index_mapper: Dict[str, int],
            readout_id
    ):
        recursive_flag = True
        if isinstance(value, str):
            recursive_flag = False
            temp_value = [v for v in jieba.cut(value)]
            # 修改之前插入的value
            graph.value[preds[-1]] = []
        else:
            temp_value = value

        if len(temp_value) == 1 and len(preds) > 0:
            graph.value[preds[-1]] = temp_value[0]
            return

        node_ids = []
        for v in temp_value:
            node_id = graph.insert_node(component_id, name, v, self_loop=self.self_loop, readout_id=readout_id)

            new_preds = preds[-self.list_pred_degree:]
            if self.list_inner_degree >= 0:
                new_preds += node_ids[-self.list_inner_degree:]

            new_succs = [node_id] * len(new_preds)
            if self.r_list_inner_degree >= 0:
                new_succs += node_ids[-self.r_list_inner_degree:]
                new_preds += [node_id] * (len(new_succs) - len(new_preds))

            graph.insert_edges(new_preds, new_succs)
            node_ids.append(node_id)
            if recursive_flag:
                self.insert_component(
                    graph=graph, component_id=component_id, name=name, value=v, preds=preds + [node_id],
                    local_index_mapper=local_index_mapper, global_index_mapper=global_index_mapper,
                    readout_id=readout_id
                )

    def insert_component(
            self,
            graph: Graph,
            component_id: int,
            name: str,
            value: Union[Dict, List, str, int, float, None],
            preds: Union[List[int], None],
            local_index_mapper: Dict[str, int],
            global_index_mapper: Dict[str, int],
            readout_id: Union[int, None]
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
        :param readout_id:
        :return:
        """
        if preds is None:
            node_id = graph.insert_node(component_id, name, value, self_loop=self.self_loop, readout_id=readout_id)
            self.insert_component(
                graph=graph, component_id=component_id, name=name, value=value, preds=[],
                local_index_mapper=local_index_mapper, global_index_mapper=global_index_mapper, readout_id=node_id
            )
        elif isinstance(value, dict):
            self.insert_dict(
                graph=graph, component_id=component_id, value=value, preds=preds,
                local_index_mapper=local_index_mapper, global_index_mapper=global_index_mapper, readout_id=readout_id
            )
        elif isinstance(value, list) or (self.segmentation and isinstance(value, str)):
            self.insert_array(
                graph=graph, component_id=component_id, name=name, value=value, preds=preds,
                local_index_mapper=local_index_mapper, global_index_mapper=global_index_mapper, readout_id=readout_id
            )

    def parse(
            self,
            jsons: Iterable[Union[Dict, List]],
            progress_bar: bool = False,
    ) -> (Graph, dict, List[dict]):
        graph = Graph()
        global_index_mapper = {}
        local_index_mappers = []
        jsons = progress_wrapper(jsons, disable=not progress_bar, postfix='parsing json')
        for i, value in enumerate(jsons):
            local_index_mapper = {}
            self.insert_component(
                graph=graph, component_id=i+1, name=READOUT, value=value, preds=None,
                local_index_mapper=local_index_mapper, global_index_mapper=global_index_mapper, readout_id=None
            )
            local_index_mappers.append(local_index_mapper)
        return graph, global_index_mapper, local_index_mappers
