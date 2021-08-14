import os
import json
import numpy as np
import pandas as pd
from all2graph.graph import JsonGraph
from toad.utils.progress import Progress


def test_flatten_dict():
    inputs = {
        'a': {
            'b': {
                'c': 'd',
                'e': 'f'
            },
        }
    }
    jg1 = JsonGraph()
    jg1.insert_component(0, 'inputs', inputs)
    assert jg1.num_nodes == 5 and jg1.num_edges == 4

    jg2 = JsonGraph(flatten_dict=True)
    jg2.insert_component(0, 'inputs', inputs)
    assert jg2.num_nodes == 3 and jg2.num_edges == 2


def test_list_pred_degree():
    inputs = {
        'a': {
            'b': [1, 2, 3, 4],
        }
    }
    jg1 = JsonGraph()
    jg1.insert_component(0, 'inputs', inputs)
    assert jg1.num_nodes == 7 and jg1.num_edges == 6

    jg2 = JsonGraph(list_pred_degree=0)
    jg2.insert_component(0, 'inputs', inputs)
    assert jg2.num_nodes == 7 and jg2.num_edges == 14, '\n{}\n{}\n{}\n{}'.format(
        jg2.names, jg2.values, jg2.preds, jg2.succs
    )


def test_list_inner_degree():
    inputs = {
        'a': [
            [1, 2, 3],
            [4, 5, 6]
        ]
    }
    jg1 = JsonGraph()
    jg1.insert_component(0, 'inputs', inputs)
    assert jg1.num_nodes == 10 and jg1.num_edges == 9

    jg2 = JsonGraph(list_inner_degree=0)
    jg2.insert_component(0, 'inputs', inputs)
    assert jg2.num_nodes == 10 and jg2.num_edges == 16, '\n{}\n{}\n{}\n{}'.format(
        jg2.names, jg2.values, jg2.preds, jg2.succs
    )

    jg2 = JsonGraph(list_inner_degree=0, r_list_inner_degree=1)
    jg2.insert_component(0, 'inputs', inputs)
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
    jg1 = JsonGraph(flatten_dict=True, dict_pred_degree=0, list_pred_degree=0, list_inner_degree=2,
                    r_list_inner_degree=1)
    jg1.insert_component(0, 'inputs', inputs)
    assert jg1.num_nodes == 10 and jg1.num_edges == 34, '\n{}\n{}\n{}\n{}'.format(
        jg1.names, jg1.values, jg1.preds, jg1.succs
    )


def test_json_graph():
    path = os.path.dirname(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
    df = pd.read_csv(path)
    json_graph = JsonGraph(flatten_dict=True)
    for i, value in enumerate(Progress(df.json.values)):
        json_graph.insert_component(i, 'graph', json.loads(value))
    assert json_graph.num_nodes - json_graph.num_edges == df.shape[0]
    assert np.unique(json_graph.component_ids).shape[0] == df.shape[0]

    json_graph = JsonGraph(dict_pred_degree=0, list_pred_degree=0, list_inner_degree=0, r_list_inner_degree=0)
    index_mapper = {}
    for i, value in enumerate(Progress(df.json.values)):
        json_graph.insert_component(
            component_id=i,
            name='graph',
            value=json.loads(value),
            index_names={
                'name'
            },
            index_mapper=index_mapper
        )
    assert len(index_mapper) > 0
    assert np.unique(json_graph.component_ids).shape[0] == df.shape[0]
    print(json_graph.num_nodes, json_graph.num_edges)
    print(np.unique(json_graph.names))


if __name__ == '__main__':
    test_flatten_dict()
    test_list_pred_degree()
    test_list_inner_degree()
    test_complicated_situation()
    test_json_graph()
