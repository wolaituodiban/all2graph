import os

import pandas as pd
import all2graph as ag
import matplotlib.pyplot as plt
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
    graph, global_index_mapper, local_index_mappers = parser.parse(node_df, disable=False)

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
    s1 = ag.graph.RawGraph(
        component_id=[0, 0], key=['a', 'b'], value=['a', 'b'], src=[1], dst=[0], symbol=['readout', 'value'])
    s2 = ag.graph.RawGraph(
        component_id=[0, 0], key=['a', 'b'], value=['a', 'c'], src=[1], dst=[0], symbol=['readout', 'value'])
    s3 = ag.graph.RawGraph.batch([s1, s2])

    m1 = ag.MetaInfo.from_data(s1)
    m2 = ag.MetaInfo.from_data(s2)
    m3 = ag.MetaInfo.reduce([m1, m2])
    m4 = ag.MetaInfo.from_data(s3)
    assert m3.__eq__(m4, debug=True)


def test_json():
    import json
    s1 = ag.graph.RawGraph(
        component_id=[0, 0], key=['a', 'b'], value=['a', 'b'], src=[1], dst=[0], symbol=['readout', 'value'])
    temp = json.dumps(s1.to_json())
    temp = json.loads(temp)
    s2 = ag.graph.RawGraph.from_json(temp)
    assert s1 == s2


def test_filter_node():
    s1 = ag.graph.RawGraph(
        component_id=[0, 0, 0], key=['a', 'b', 'c'], value=['e', 'f', 'd'], src=[1, 1, 2, 0], dst=[0, 1, 1, 1],
        symbol=['readout', 'value', 'value']
    )
    s2, _ = s1.filter_node({'b', 'c'})
    assert s2 == ag.graph.RawGraph(
        component_id=[0, 0], key=['b', 'c'], value=['f', 'd'], src=[0, 1], dst=[0, 0], symbol=['value', 'value']
    ), s2.to_json()


def test_filter_edge():
    s1 = ag.graph.RawGraph(
        component_id=[0, 0, 0], key=['a', 'b', 'c'], value=['e', 'f', 'd'], src=[1, 1, 2, 0], dst=[0, 1, 1, 1],
        symbol=['readout', 'value', 'value']
    )
    s2, _ = s1.filter_edge({('c', 'b')})
    assert s2 == ag.graph.RawGraph(
        component_id=[0, 0, 0], key=['a', 'b', 'c'], value=['e', 'f', 'd'], symbol=['readout', 'value', 'value'],
        src=[2], dst=[1],
    ), s2.to_json()


def test_draw():
    import json
    data = {
        'SMALL_LOAN': [
            {
                'ord_no': 'CH202007281033864',
                'bsy_typ': 'CASH',
                'prc_amt': 3600.0,
                'crt_tim': '2020-07-28 16:54:31',
                'adt_lmt': 3600.0,
                'avb_lmt': 0.0,
                'avb_lmt_rat': 0.0
            },
            {
                'ord_no': 'CH202007281033864',
                'bsy_typ': 'CASH',
                'stg_no': '1',
                'rep_dte': '2020-08-28',
                'rep_tim': '2020-08-28 08:35:05',
                'prc_amt': -286.93,
                'ded_typ': 'AUTO_DEDUCT',
                'adt_lmt': 3600.0,
                'avb_lmt': 286.93,
                'avb_lmt_rat': 0.079703
            },
            {
                'ord_no': 'CH202007281033864',
                'bsy_typ': 'CASH',
                'stg_no': '2',
                'rep_dte': '2020-09-28',
                'rep_tim': '2020-09-28 10:17:18',
                'prc_amt': -289.15,
                'ded_typ': 'MANUAL_REPAY',
                'adt_lmt': 3600.0,
                'avb_lmt': 576.08,
                'avb_lmt_rat': 0.160022
            }
        ]
    }
    data = pd.DataFrame(
        {
            'json': [json.dumps(data)],
            'crt_dte': '2020-10-09'
        }
    )

    parser = JsonParser('json', local_id_keys={'ord_no'}, list_inner_degree=1, self_loop=False)
    graph, global_index_mapper, local_index_mappers = parser.parse(data, disable=False)
    graph.draw(disable=False)
    plt.show()
    graph.draw(disable=False, exclude_keys={'avb_lmt'}, include_keys={'readout', 'SMALL_LOAN', 'avb_lmt'})
    plt.show()


if __name__ == '__main__':
    test_meta_graph()
    test_batch()
    test_filter_node()
    test_filter_edge()
    test_draw()
