import os
import json
import numpy as np
import pandas as pd
from all2graph.json import JsonParser
from all2graph import JiebaTokenizer


def test_flatten_dict():
    inputs = {
        'a': {
            'b': {
                'c': 'd',
                'e': 'f'
            },
        }
    }
    inputs = pd.DataFrame([json.dumps(inputs)], columns=['json'])
    jg1, *_ = JsonParser('json').parse(inputs)
    assert jg1.num_nodes == 5 and jg1.num_edges == 4, '{}\n{}'.format(jg1.src, jg1.dst)

    jg2, *_ = JsonParser('json', flatten_dict=True).parse(inputs)
    assert jg2.num_nodes == 3 and jg2.num_edges == 2


def test_list_dst_degree():
    inputs = {
        'a': {
            'b': [1, 2, 3, 4],
        }
    }
    inputs = pd.DataFrame([json.dumps(inputs)], columns=['json'])
    jg1, *_ = JsonParser('json').parse(inputs)
    assert jg1.num_nodes == 7 and jg1.num_edges == 6

    jg2, *_ = JsonParser('json', list_dst_degree=0).parse(inputs)
    assert jg2.num_nodes == 7 and jg2.num_edges == 14, '\n{}\n{}\n{}\n{}'.format(
        jg2.key, jg2.value, jg2.src, jg2.dst
    )


def test_list_inner_degree():
    inputs = {
        'a': [
            [1, 2, 3],
            [4, 5, 6]
        ]
    }
    inputs = pd.DataFrame([json.dumps(inputs)], columns=['json'])
    jg1, *_ = JsonParser('json').parse(inputs)
    assert jg1.num_nodes == 10 and jg1.num_edges == 9

    jg2, *_ = JsonParser('json', list_inner_degree=0).parse(inputs)
    assert jg2.num_nodes == 10 and jg2.num_edges == 16, '\n{}\n{}\n{}\n{}'.format(
        jg2.key, jg2.value, jg2.src, jg2.dst)

    jg2, *_ = JsonParser('json', list_inner_degree=0, r_list_inner_degree=1).parse(inputs)
    assert jg2.num_nodes == 10 and jg2.num_edges == 21, '\n{}\n{}\n{}\n{}'.format(
        jg2.key, jg2.value, jg2.src, jg2.dst)


def test_complicated_situation():
    inputs = {
        'a': {
            'b': [
                [{'c': 'd'}],
                [1, 2, 3, 4]
            ]
        }
    }
    inputs = pd.DataFrame([json.dumps(inputs)], columns=['json'])
    jg1, *_ = JsonParser('json', flatten_dict=True, dict_dst_degree=0, list_dst_degree=0, list_inner_degree=2,
                         r_list_inner_degree=1, target_cols=['a', 'b']).parse(inputs)
    assert jg1.num_nodes == 10 and jg1.num_edges == 34, '\n{}\n{}\n{}\n{}'.format(
        jg1.key, jg1.value, jg1.src, jg1.dst
    )


def speed():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    df = pd.read_csv(path)
    json_graph, *_ = JsonParser('json', flatten_dict=True).parse(
        df, progress_bar=True
    )
    # assert json_graph.num_nodes - json_graph.num_edges == df.shape[0]
    assert np.unique(np.abs(json_graph.component_id)).shape[0] == df.shape[0]
    print(json_graph.num_nodes, json_graph.num_edges)
    print(np.unique(json_graph.key))
    print(max(map(len, filter(lambda x: isinstance(x, str), json_graph.value))))

    json_graph2, global_index_mapper, _ = JsonParser(
        'json', dict_dst_degree=0, list_dst_degree=0, list_inner_degree=0, r_list_inner_degree=0,
        global_id_keys={'name'}, segment_value=False, self_loop=True
    ).parse(df, progress_bar=True)
    assert len(global_index_mapper) > 0
    assert np.unique(np.abs(json_graph2.component_id)).shape[0] == df.shape[0]
    assert json_graph2.num_nodes < json_graph.num_nodes
    print(json_graph2.num_nodes, json_graph2.num_edges)
    print(np.unique(json_graph2.key))
    print(max(map(len, filter(lambda x: isinstance(x, str), json_graph2.value))))

    json_graph3, global_index_mapper, _ = JsonParser(
        'json', flatten_dict=True, global_id_keys={'name'}, segment_value=True, self_loop=True,
        tokenizer=JiebaTokenizer()).parse(df, progress_bar=True)

    assert len(global_index_mapper) > 0
    assert np.unique(np.abs(json_graph2.component_id)).shape[0] == df.shape[0]
    assert json_graph2.num_nodes < json_graph3.num_nodes
    print(json_graph3.num_nodes, json_graph3.num_edges)
    print(max(map(len, filter(lambda x: isinstance(x, str), json_graph3.value))))


def test_repr():
    import all2graph as ag
    parser = JsonParser(
        'json', flatten_dict=True, global_id_keys={'name'}, segment_value=True, self_loop=True,
        tokenizer=JiebaTokenizer(), processors=[
            ('$.SMALL_LOAN',),
            ('$.*', ag.Timestamp('crt_tim', '%Y-%m-%d %H:%M:%S', ['day', 'hour', 'weekday'])),
            ('$.*', ag.Timestamp('rep_tim', '%Y-%m-%d %H:%M:%S', ['day', 'hour', 'weekday'])),
            ('$.*', ag.Timestamp('rep_dte', '%Y-%m-%d', ['day', 'weekday'])),
            ('$.*.bsy_typ', ag.Lower()),
            ('$.*.ded_typ', ag.Lower()),
            ('$.*.bsy_typ', ag.Split('_')),
            ('$.*.ded_typ', ag.Split('_')),
            ('$.*', ag.Delete(['crt_tim', 'rep_tim', 'rep_dte', 'prc_amt', 'adt_lmt', 'avb_lmt']))
        ]
    )
    print(parser)


if __name__ == '__main__':
    test_flatten_dict()
    test_list_dst_degree()
    test_list_inner_degree()
    test_complicated_situation()
    test_repr()
    speed()
