import json
from typing import Dict, Type, Tuple

import networkx as nx

from ..macro import TYPE, NODES, EDGES, SEP
from ..meta_struct import MetaStruct
from ..meta_edge import MetaEdge, ALL_EDGE_CLASSES
from ..meta_node import MetaNode, ALL_NODE_CLASSES


ALL_NODE_EDGE_CLASSES = {}
ALL_NODE_EDGE_CLASSES.update(ALL_NODE_CLASSES)
ALL_NODE_EDGE_CLASSES.update(ALL_EDGE_CLASSES)


class MetaGraph(MetaStruct):
    """图的基类，定义基本成员变量和基本方法"""
    def __init__(self, nodes: Dict[str, MetaNode], edges: Dict[Tuple[str, str], MetaEdge], **kwargs):
        """

        :param nodes:
        :param edges:
        """
        assert len({n.num_samples for n in nodes.values()}.union(e.num_samples for e in edges.values())) == 1
        super().__init__(**kwargs)
        self.nodes = nodes
        self.edges = edges
        # 检查是否存在孤立点
        self.to_networkx()

    def __eq__(self, other):
        return super().__eq__(other) and self.nodes == other.nodes and self.edges == other.edges

    def to_json(self) -> dict:
        output = super().to_json()
        output.update({
            NODES: {k: v.to_json() for k, v in self.nodes.items()},
            EDGES: {SEP.join(k): v.to_json() for k, v in self.edges.items()}
        })
        return output

    @classmethod
    def from_json(cls, obj, classes: Dict[str, Type[MetaNode]] = None):
        """

        :param obj: json对象
        :param classes: 所有json中包含的的点和边的类，如果为空，那么取默认
        :return:
        """
        if isinstance(obj, str):
            obj = json.loads(obj)
        else:
            obj = dict(obj)

        if classes is None:
            classes = ALL_NODE_EDGE_CLASSES
        else:
            all_node_edge_classes = dict(ALL_NODE_EDGE_CLASSES)
            all_node_edge_classes.update(classes)
            classes = all_node_edge_classes

        obj[NODES] = {k: classes[v[TYPE]].from_json(v) for k, v in obj[NODES].items()}
        obj[EDGES] = {tuple(k.split(SEP)): classes[v[TYPE]].from_json(v) for k, v in obj[EDGES].items()}
        return super().from_json(obj)

    @classmethod
    def reduce(cls, graphs, **kwargs):
        raise NotImplementedError

    def to_networkx(self) -> nx.DiGraph:
        """将对象转化成一个networkx有向图"""
        graph = nx.DiGraph()
        for name, node in self.nodes.items():
            graph.add_node(name, **node.to_json())
        for (pred, succ), edge in self.edges.items():
            graph.add_edge(pred, succ, **edge.to_json())
        assert nx.number_of_isolates(graph) == 0, "图存在孤立点"
        return graph
