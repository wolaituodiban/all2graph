import json
from all2graph.meta_graph import MetaNode, MetaGraph


class Node(MetaNode):
    def to_json(self) -> dict:
        return super().to_json()

    @classmethod
    def from_array(cls, x, *args, **kwargs):
        pass


class Graph(MetaGraph):
    def to_json(self) -> dict:
        return super().to_json()


def test1():
    node1 = Node('node1')
    node2 = Node('node1')
    try:
        Graph([node1, node2])
        raise ValueError('节点重名测试失败')
    except AssertionError:
        print('节点重名测试成功')


def test2():
    node1 = Node('node1', succs=['node3'])
    try:
        Graph([node1])
        raise ValueError('前置后置节点属性检测测试失败')
    except AssertionError:
        print('前置后置节点属性检测测试成功')


def test3():
    node1 = Node('node1', succs=['node2'])
    node2 = Node('node2', preds=['node3'])
    node3 = Node('node3')
    graph = Graph([node1, node2, node3])
    print(json.dumps(graph.to_json(), indent=4))


def test4():
    node1 = Node('test', preds=['a', 'b'])
    config1 = json.dumps(node1.to_json())
    node2 = Node.from_json(config1)
    config2 = json.dumps(node2.to_json())
    assert config1 == config2, 'json导入导出一致性测试失败'
    print('json导入导出一致性测试成功')


if __name__ == '__main__':
    test1()
    test2()
    test3()
    test4()
