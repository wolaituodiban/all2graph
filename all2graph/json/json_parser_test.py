import os
import json
import numpy as np
import pandas as pd
from all2graph.json import JsonParser


def test_flatten_dict():
    inputs = {
        'a': {
            'b': {
                'c': 'd',
                'e': 'f'
            },
        }
    }
    jg1, *_ = JsonParser('input', ).parse([inputs])
    assert jg1.num_nodes == 5 and jg1.num_edges == 4

    jg2, *_ = JsonParser('input', flatten_dict=True).parse([inputs])
    assert jg2.num_nodes == 3 and jg2.num_edges == 2


def test_list_pred_degree():
    inputs = {
        'a': {
            'b': [1, 2, 3, 4],
        }
    }
    jg1, *_ = JsonParser('input', ).parse([inputs])
    assert jg1.num_nodes == 7 and jg1.num_edges == 6

    jg2, *_ = JsonParser('input', list_pred_degree=0).parse([inputs])
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
    jg1, *_ = JsonParser('input', ).parse([inputs])
    assert jg1.num_nodes == 10 and jg1.num_edges == 9

    jg2, *_ = JsonParser('input', list_inner_degree=0).parse([inputs])
    assert jg2.num_nodes == 10 and jg2.num_edges == 16, '\n{}\n{}\n{}\n{}'.format(
        jg2.key, jg2.value, jg2.src, jg2.dst
    )

    jg2, *_ = JsonParser('input', list_inner_degree=0, r_list_inner_degree=1).parse([inputs])
    assert jg2.num_nodes == 10 and jg2.num_edges == 21


def test_complicated_situation():
    inputs = {
        'a': {
            'b': [
                [{'c': 'd'}],
                [1, 2, 3, 4]
            ]
        }
    }
    jg1, *_ = JsonParser('input', flatten_dict=True, dict_pred_degree=0, list_pred_degree=0, list_inner_degree=2,
                         r_list_inner_degree=1).parse([inputs])
    assert jg1.num_nodes == 8 and jg1.num_edges == 27, '\n{}\n{}\n{}\n{}'.format(
        jg1.key, jg1.value, jg1.src, jg1.dst
    )


def speed():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    df = pd.read_csv(path)
    json_graph, *_ = JsonParser(root_name='graph', flatten_dict=True).parse(
        list(map(json.loads, df.json)), progress_bar=True
    )
    assert json_graph.num_nodes - json_graph.num_edges == df.shape[0]
    assert np.unique(json_graph.component_id).shape[0] == df.shape[0]
    print(json_graph.num_nodes, json_graph.num_edges)
    print(np.unique(json_graph.key))
    print(max(map(len, filter(lambda x: isinstance(x, str), json_graph.value))))

    json_graph2, global_index_mapper, _ = JsonParser(
        dict_pred_degree=0, list_pred_degree=0, list_inner_degree=0, r_list_inner_degree=0, global_index_names={'name'},
        segmentation=False, root_name='graph', self_loop=True
    ).parse(list(map(json.loads, df.json)), progress_bar=True)
    assert len(global_index_mapper) > 0
    assert np.unique(json_graph2.component_id).shape[0] == df.shape[0]
    assert json_graph2.num_nodes < json_graph.num_nodes
    print(json_graph2.num_nodes, json_graph2.num_edges)
    print(np.unique(json_graph2.key))
    print(max(map(len, filter(lambda x: isinstance(x, str), json_graph2.value))))

    json_graph3, global_index_mapper, _ = JsonParser(
        flatten_dict=True, global_index_names={'name'}, segmentation=True, root_name='graph', self_loop=True
    ).parse(list(map(json.loads, df.json)), progress_bar=True)

    assert len(global_index_mapper) > 0
    assert np.unique(json_graph2.component_id).shape[0] == df.shape[0]
    assert json_graph2.num_nodes < json_graph3.num_nodes
    print(json_graph3.num_nodes, json_graph3.num_edges)
    print(max(map(len, filter(lambda x: isinstance(x, str), json_graph3.value))))


if __name__ == '__main__':
    test_flatten_dict()
    test_list_pred_degree()
    test_list_inner_degree()
    test_complicated_situation()
    speed()
