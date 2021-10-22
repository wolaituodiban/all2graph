import os
import pandas as pd
import all2graph as ag
from all2graph.json import JsonParser


def test_meta_graph():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    csv_path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    node_df = pd.read_csv(csv_path, nrows=64)

    parser = JsonParser(
        'json', flatten_dict=True, local_id_keys={'name'}, segment_value=True, self_loop=True,
        list_inner_degree=1
    )
    graph, global_index_mapper, local_index_mappers = parser.parse(
        node_df, progress_bar=True
    )

    meta_graph, meta_node_id, meta_edge_id = graph.meta_graph()
    assert graph.key == [meta_graph.value[i] for i in meta_node_id]
    assert graph.component_id == [meta_graph.component_id[i] for i in meta_node_id]
    assert [graph.key[i] for i in graph.src] == [meta_graph.value[meta_graph.src[i]] for i in meta_edge_id]
    assert [graph.key[i] for i in graph.dst] == [meta_graph.value[meta_graph.dst[i]] for i in meta_edge_id]
    print(graph)


def test_batch():
    # 测试思路
    # 1、构造两个样本s1和s2
    # 2、分别使用s1和s2构造两个元数据对象m1和m2
    # 3、合并m1和m2得到m3
    # 4、将s1和s2合并得到s3
    # 5、使用s3构造元数据对象m4
    # 如果代码正确，那么m3和m4应该相等
    s1 = ag.RawGraph(
        component_id=[0, 0], key=['a', 'b'], value=['a', 'b'], src=[1], dst=[0], symbol=['readout', 'value'])
    s2 = ag.RawGraph(
        component_id=[0, 0], key=['a', 'b'], value=['a', 'c'], src=[1], dst=[0], symbol=['readout', 'value'])
    s3 = ag.RawGraph.batch([s1, s2])

    m1 = ag.MetaInfo.from_data(s1)
    m2 = ag.MetaInfo.from_data(s2)
    m3 = ag.MetaInfo.reduce([m1, m2])
    m4 = ag.MetaInfo.from_data(s3)
    assert m3.__eq__(m4, debug=True)


def test_json():
    import json
    s1 = ag.RawGraph(
        component_id=[0, 0], key=['a', 'b'], value=['a', 'b'], src=[1], dst=[0], symbol=['readout', 'value'])
    temp = json.dumps(s1.to_json())
    temp = json.loads(temp)
    s2 = ag.RawGraph.from_json(temp)
    assert s1 == s2


if __name__ == '__main__':
    test_meta_graph()
    test_batch()
