from typing import Dict, List, Union


class Graph:
    def __init__(self):
        self.component_ids: List[int] = []
        self.names: List[str] = []
        self.values: List[Union[Dict, List, str, int, float, None]] = []
        self.preds: List[int] = []
        self.succs: List[int] = []

    @property
    def num_nodes(self):
        assert len(self.names) == len(self.values)
        return len(self.names)

    @property
    def num_edges(self):
        assert len(self.preds) == len(self.succs)
        return len(self.preds)

    def insert_edges(self, preds: List[int], succs: List[int]):
        self.preds += preds
        self.succs += succs

    def insert_node(
            self,
            patch_id: int,
            name: str,
            value: Union[Dict, List, str, int, float, None],
    ) -> int:
        node_id = len(self.names)
        self.component_ids.append(patch_id)
        self.names.append(name)
        self.values.append(value)
        return node_id

    def insert_component(
            self,
            component_id: int,
            name: str,
            value: Union[Dict, List, str, int, float, None],
            preds: List[int] = None,
    ):
        """
        插入一个连通片（component）。如果图中任意两点都是连通的，那么图被称作连通图。
        :param component_id: 连通片编号
        :param name: 第一个节点的名称
        :param value: 第一个节点的值
        :param preds: 前置节点的编号
        :return:
        """
        raise NotImplementedError
