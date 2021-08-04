import json
from all2graph.meta_graph import MetaEdge, MetaNode, MetaGraph


class Node(MetaNode):
    def to_json(self) -> dict:
        return super().to_json()

    @classmethod
    def from_json(cls, obj):
        return super().from_json(obj)

    @classmethod
    def from_data(cls, x, *args, **kwargs):
        pass


class Graph(MetaGraph):
    @classmethod
    def from_data(cls, **kwargs):
        pass


def test1():
    edge = MetaEdge([1, 2], [0.5, 1], 2)
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
    json_obj = graph.to_json()
    graph = Graph.from_json(json_obj, {'Node': Node, "MetaEdge": MetaEdge})
    json_obj2 = graph.to_json()
    assert json_obj2 == json_obj, 'json导入导出一致性测试失败'
    print(json.dumps(json_obj2, indent=2))
    print('json导入导出一致性测试成功')


if __name__ == '__main__':
    test1()
    test2()
    print('测试MetaGraph成功')
