import json
import pandas as pd
import jsonpromax as jpm
from all2graph.json import JsonParser
from all2graph import JiebaTokenizer


def test_list_dst_degree():
    inputs = {
        'a': {
            'b': [1, 2, 3, 4],
        }
    }
    inputs = pd.DataFrame([json.dumps(inputs)], columns=['json'])
    inputs['day'] = None
    jg1, *_ = JsonParser('json', time_col='day', self_loop=False, list_inner_degree=0).parse(inputs)
    assert jg1.num_nodes == 7, jg1.value
    assert jg1.num_edges == 6, jg1.to_df('value')

    jg2, *_ = JsonParser('json', time_col='day', list_dst_degree=0, self_loop=False, list_inner_degree=0).parse(inputs)
    assert jg2.num_nodes == 7 and jg2.num_edges == 14, jg2.to_df('value')


def test_list_inner_degree():
    inputs = {
        'a': [
            [1, 2, 3],
            [4, 5, 6]
        ]
    }
    inputs = pd.DataFrame([json.dumps(inputs)], columns=['json'])
    inputs['day'] = None
    jg1, *_ = JsonParser('json', time_col='day', self_loop=False, list_inner_degree=0).parse(inputs)
    assert jg1.num_nodes == 10 and jg1.num_edges == 9, jg1.to_df('value')

    jg2, *_ = JsonParser('json', time_col='day', list_inner_degree=-1, self_loop=False).parse(inputs)
    assert jg2.num_nodes == 10 and jg2.num_edges == 16, jg2.to_df('value')

    jg2, *_ = JsonParser(
        'json', time_col='day', list_inner_degree=-1, r_list_inner_degree=1, self_loop=False).parse(inputs)
    assert jg2.num_nodes == 10 and jg2.num_edges == 21, jg2.to_df('value')


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
    inputs['day'] = None
    jg1, *_ = JsonParser(
        'json', time_col='day', dict_dst_degree=0, list_dst_degree=0,
        list_inner_degree=2, r_list_inner_degree=1, self_loop=False).parse(inputs)
    assert jg1.num_nodes == 11 and jg1.num_edges == 44, jg1.to_df('value')


def test_repr():
    import all2graph as ag
    parser = JsonParser(
        'json', time_col='day', global_id_keys={'name'}, self_loop=True, tokenizer=JiebaTokenizer(),
        processor=ag.json.JsonPathTree([
            ('$.SMALL_LOAN',),
            ('$.*', ag.json.Timestamp('crt_tim', '%Y-%m-%d %H:%M:%S', ['day', 'hour', 'weekday'])),
            ('$.*', ag.json.Timestamp('rep_tim', '%Y-%m-%d %H:%M:%S', ['day', 'hour', 'weekday'])),
            ('$.*', ag.json.Timestamp('rep_dte', '%Y-%m-%d', ['day', 'weekday'])),
            ('$.*.bsy_typ', ag.json.Lower()),
            ('$.*.ded_typ', ag.json.Lower()),
            ('$.*.bsy_typ', ag.json.Split('_')),
            ('$.*.ded_typ', ag.json.Split('_')),
            ('$.*', jpm.Delete(['crt_tim', 'rep_tim', 'rep_dte', 'prc_amt', 'adt_lmt', 'avb_lmt']))
        ])
    )
    print(parser)


def test_grid():
    inputs = [{'a': 1}, {'a': 2}]
    inputs = pd.DataFrame([json.dumps(inputs)], columns=['json'])
    inputs['day'] = None
    jg1, *_ = JsonParser('json', time_col='day', grid=False, self_loop=False).parse(inputs)
    jg2, *_ = JsonParser('json', time_col='day', grid=True, self_loop=False).parse(inputs)
    assert jg2.num_edges - jg1.num_edges == 1
    jg2.draw()
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == '__main__':
    test_list_dst_degree()
    test_list_inner_degree()
    test_complicated_situation()
    test_repr()
    test_grid()
