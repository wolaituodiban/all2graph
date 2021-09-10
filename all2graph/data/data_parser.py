from typing import Iterable, Union, List, Dict
from ..graph import Graph


class DataParser:
    def __init__(self, **kwargs):
        pass

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
        raise NotImplementedError

    def parse(self, data: Iterable, progress_bar: bool = False):
        raise NotImplementedError

    __call__ = parse
