import json
from typing import Dict, Type, Tuple

import networkx as nx

from ..macro import TYPE, NODES, EDGES, SEP
from ..meta_struct import MetaStruct
from .meta_edge import MetaEdge
from .meta_node import MetaNode


class MetaGraph(MetaStruct):
    # todo 节点和边都改为用字典存储
    """图的基类，定义基本成员变量和基本方法"""
    def __init__(self, nodes: Dict[str, MetaNode], edges: Dict[Tuple[str, str], MetaEdge], **kwargs):
        """

        :param nodes:
        :param edges:
        """
        super().__init__(**kwargs)
        self.nodes = nodes
        self.edges = edges
        # 检查是否存在孤立点
        self.to_networkx()

    def to_json(self) -> dict:
        output = super().to_json()
        output.update({
            NODES: {k: v.to_json() for k, v in self.nodes.items()},
            EDGES: {SEP.join(k): v.to_json() for k, v in self.edges.items()}
        })
        return output

    @classmethod
    def from_json(cls, obj, cls_dict: Dict[str, Type[MetaNode]]):
        if isinstance(obj, str):
            obj = json.loads(obj)
        else:
            obj = dict(obj)
        obj[NODES] = {k: cls_dict[v[TYPE]].from_json(v) for k, v in obj[NODES].items()}
        obj[EDGES] = {tuple(k.split(SEP)): cls_dict[v[TYPE]].from_json(v) for k, v in obj[EDGES].items()}
        return super().from_json(obj)

    def to_networkx(self) -> nx.DiGraph:
        """将对象转化成一个networkx有向图"""
        graph = nx.DiGraph()
        for name, node in self.nodes.items():
            graph.add_node(name, **node.to_json())
        for (pred, succ), edge in self.edges.items():
            graph.add_edge(pred, succ, **edge.to_json())
        assert nx.number_of_isolates(graph) == 0, "图存在孤立点"
        return graph
