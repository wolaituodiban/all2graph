import json
from typing import Dict, Type, Tuple

import networkx as nx

from ..graph import Graph
from ..meta_struct import MetaStruct
from .meta_edge import MetaEdge, ALL_EDGE_CLASSES
from .meta_node import MetaNode, ALL_NODE_CLASSES


ALL_NODE_EDGE_CLASSES = {}
ALL_NODE_EDGE_CLASSES.update(ALL_NODE_CLASSES)
ALL_NODE_EDGE_CLASSES.update(ALL_EDGE_CLASSES)


class MetaGraph(MetaStruct):
    SEP = ','
    NODES = 'nodes'
    EDGES = 'edges'
    """图的基类，定义基本成员变量和基本方法"""
    def __init__(self, nodes: Dict[str, MetaNode], edges: Dict[Tuple[str, str], MetaEdge], **kwargs):
        """

        :param nodes:
        :param edges:
        """
        assert len({n.num_samples for n in nodes.values()}) == 1, {k: n.num_samples for k, n in nodes.items()}
        assert len({e.num_samples for e in edges.values()}) == 1, {k: e.num_samples for k, e in edges.items()}
        assert list({n.num_samples for n in nodes.values()}) == list({e.num_samples for e in edges.values()})
        super().__init__(**kwargs)
        self.nodes = nodes
        self.edges = edges
        # 检查是否存在孤立点
        self.to_networkx()

    def __eq__(self, other):
        return super().__eq__(other) and self.nodes == other.nodes and self.edges == other.edges

    @property
    def num_samples(self):
        return self.nodes[list(self.nodes)[0]].num_samples

    def to_json(self) -> dict:
        output = super().to_json()
        output.update({
            self.NODES: {k: v.to_json() for k, v in self.nodes.items()},
            self.EDGES: {self.SEP.join(k): v.to_json() for k, v in self.edges.items()}
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

        obj[cls.NODES] = {k: classes[v[cls.TYPE]].from_json(v) for k, v in obj[cls.NODES].items()}
        obj[cls.EDGES] = {tuple(k.split(cls.SEP)): classes[v[cls.TYPE]].from_json(v) for k, v in obj[cls.EDGES].items()}
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

    def create_graph(self, **kwargs) -> Graph:
        raise NotImplementedError

