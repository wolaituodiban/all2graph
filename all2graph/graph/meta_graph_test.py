import json
from all2graph.graph import MetaGraph
from all2graph.node import MetaNode
from all2graph.edge import MetaEdge


class Node(MetaNode):
    def to_json(self) -> dict:
        return super().to_json()

    def __eq__(self, other):
        return super().__eq__(other)

    @classmethod
    def from_json(cls, obj):
        return super().from_json(obj)

    @classmethod
    def from_data(cls, x, *args, **kwargs):
        pass

    @classmethod
    def merge(cls, **kwargs):
        raise NotImplementedError


class Edge(MetaEdge):
    def __eq__(self, other):
        return super().__eq__(other)


class Graph(MetaGraph):
    @classmethod
    def from_data(cls, **kwargs):
        pass

    @classmethod
    def merge(cls, **kwargs):
        raise NotImplementedError


def test1():
    edge = Edge([1, 2], [0.5, 1], 2)
    try:
        Graph(
            nodes={
                'a': Node(),
                'b': Node(),
                'c': Node()
            },
            edges={
                ('a', 'b'): edge
            }
        )
        raise RuntimeError('孤立点检测失败')
    except AssertionError:
        print('孤立点检测成功')


def test2():
    graph = Graph(
        nodes={
            'a': Node(),
            'b': Node(),
            'c': Node()
        },
        edges={
            ('a', 'b'): MetaEdge([1, 2], [0.5, 1], 2),
            ('a', 'c'): MetaEdge([1, 3], [0.24, 1], 5)
        }
    )
    graph2 = Graph.from_json(json.dumps(graph.to_json()), {'Node': Node, 'Edge': Edge})
    assert graph == graph2, 'json导入导出一致性测试失败'
    print(json.dumps(graph.to_json(), indent=2))
    print('json导入导出一致性测试成功')


if __name__ == '__main__':
    test1()
    test2()
    print('测试MetaGraph成功')
