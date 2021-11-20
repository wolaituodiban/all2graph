import os
import pandas as pd

import all2graph as ag


path = os.path.dirname(__file__)
path = os.path.dirname(path)
path = os.path.dirname(path)

csv_path = os.path.join(path, 'test_data', 'MensShoePrices.csv')
df = pd.read_csv(csv_path, nrows=64)

parser = ag.json.JsonParser(
    'json', flatten_dict=True, local_id_keys={'name'}, segment_value=True, self_loop=True,
    list_inner_degree=1, tokenizer=ag.JiebaTokenizer()
)
raw_graph, global_index_mapper, local_index_mappers = parser.parse(df, disable=False)

index_ids = list(global_index_mapper.values())
for mapper in local_index_mappers:
    index_ids += list(mapper.values())
meta_info = ag.MetaInfo.from_data(raw_graph, index_nodes=index_ids, disable=False)


def test_init():
    ag.RawGraphParser.from_data(meta_info, min_df=0.01, max_df=0.95, top_k=100, top_method='max_tfidf')


def test_parse():
    trans1 = ag.RawGraphParser.from_data(meta_info)
    with ag.Timer('speed'):
        graph = trans1.parse(raw_graph)
    print(graph)


def test_eq():
    parser1 = ag.RawGraphParser(
        meta_numbers={'a': ag.MetaNumber.from_data(2, [0, 0, 1, 1, 1], [0.5, 0.2, 0.3, 0.2, -0.1])},
        strings=['a', 'b', 'c'],
        keys=['a', 'b', 'c'],
        edge_type={('a', 'b'), ('a', 'a')},
        targets=['a', 'b']
    )
    parser2 = ag.RawGraphParser(
        meta_numbers={'a': ag.MetaNumber.from_data(2, [0, 0, 1, 1, 1], [0.5, 0.2, 0.3, 0.2, -0.1])},
        strings=['b', 'a', 'c'],
        keys=['a', 'c', 'b'],
        edge_type={('a', 'a'), ('a', 'b')},
        targets=['a', 'b', ]
    )
    assert parser1 == parser2

    # meta_numbers
    parser2 = ag.RawGraphParser(
        meta_numbers={'a': ag.MetaNumber.from_data(2, [0, 0, 1, 1], [0.5, 0.2, 0.3, -0.1])},
        strings=['a', 'b', 'c'],
        keys=['a', 'b', 'c'],
        edge_type={('a', 'b'), ('a', 'a')},
        targets=['a', 'b']
    )
    assert parser1 != parser2

    # strings
    parser2 = ag.RawGraphParser(
        meta_numbers={'a': ag.MetaNumber.from_data(2, [0, 0, 1, 1, 1], [0.5, 0.2, 0.3, 0.2, -0.1])},
        strings=['a', 'b', 'cc'],
        keys=['a', 'b', 'c'],
        edge_type={('a', 'b'), ('a', 'a')},
        targets=['a', 'b']
    )
    assert parser1 != parser2

    # keys
    parser2 = ag.RawGraphParser(
        meta_numbers={'a': ag.MetaNumber.from_data(2, [0, 0, 1, 1, 1], [0.5, 0.2, 0.3, 0.2, -0.1])},
        strings=['a', 'b', 'c'],
        keys=['a', 'b', 'c', 'd'],
        edge_type={('a', 'b'), ('a', 'a')},
        targets=['a', 'b']
    )
    assert parser1 != parser2

    # edge_type
    parser2 = ag.RawGraphParser(
        meta_numbers={'a': ag.MetaNumber.from_data(2, [0, 0, 1, 1, 1], [0.5, 0.2, 0.3, 0.2, -0.1])},
        strings=['a', 'b', 'c'],
        keys=['a', 'b', 'c'],
        edge_type={('b', 'a'), ('a', 'a')},
        targets=['a', 'b']
    )
    assert parser1 != parser2

    # targets
    parser2 = ag.RawGraphParser(
        meta_numbers={'a': ag.MetaNumber.from_data(2, [0, 0, 1, 1, 1], [0.5, 0.2, 0.3, 0.2, -0.1])},
        strings=['a', 'b', 'c'],
        keys=['a', 'b', 'c'],
        edge_type={('a', 'b'), ('a', 'a')},
        targets=[]
    )
    assert parser1 != parser2


def test_reduce():
    parser1 = ag.RawGraphParser(
        meta_numbers={'a': ag.MetaNumber.from_data(2, [0, 0, 1, 1, 1], [0.5, 0.2, 0.3, 0.2, -0.1])},
        strings=['a', 'b', 'c'],
        keys=['a', 'b', 'c'],
        edge_type={('a', 'b'), ('a', 'a')},
        targets=['a', 'b']
    )

    parser2 = ag.RawGraphParser(
        meta_numbers={
            'a': ag.MetaNumber.from_data(3, [0, 0, 0, 1, 1, 2], [0.5, 0.2, 0.3, 0.2, -0.1, 0]),
            'b': ag.MetaNumber.from_data(2, [0, 1, 1], [0, 2, -3])
        },
        strings=['a', 'b', 'd'],
        keys=['a', 'b'],
        edge_type={('b', 'b'), ('a', 'b')},
        targets=[]
    )

    parser3 = ag.RawGraphParser(
        meta_numbers={
            'a': ag.MetaNumber.from_data(
                5,
                [0, 0, 1, 1, 1] + [2, 2, 2, 3, 3, 4],
                [0.5, 0.2, 0.3, 0.2, -0.1] + [0.5, 0.2, 0.3, 0.2, -0.1, 0]),
            'b': ag.MetaNumber.from_data(2, [0, 1, 1], [0, 2, -3])
        },
        strings=['a', 'b', 'c'] + ['a', 'b', 'd'],
        keys=['a', 'b', 'c'] + ['a', 'b'],
        edge_type={('a', 'b'), ('a', 'a')}.union({('b', 'b'), ('a', 'b')}),
        targets=['a', 'b']
    )

    parser4 = ag.RawGraphParser.reduce([parser1, parser2], weights=[2, 3])
    assert parser3.__eq__(parser4, True)


def test_filter_key():
    s1 = ag.graph.RawGraph(
        component_id=[0, 0, 0], key=['a', 'b', 'c'], value=['e', 'f', 'd'], src=[1, 1, 2, 0], dst=[0, 1, 1, 1],
        symbol=['readout', 'value', 'value']
    )
    s2 = ag.graph.RawGraph(
        component_id=[0, 0], key=['b', 'c'], value=['f', 'd'], src=[0, 1], dst=[0, 0], symbol=['value', 'value']
    )
    raw_graph_parser = ag.RawGraphParser.from_data(ag.MetaInfo.from_data(s2))
    assert (raw_graph_parser.parse(s1).key < 0).any()
    raw_graph_parser.set_filter_key(True)
    assert raw_graph_parser.parse(s1).value_graph.num_edges() > 0
    assert (raw_graph_parser.parse(s1).key >= 0).all()
    assert (raw_graph_parser.parse(s1).edge_key >= 0).all()


if __name__ == '__main__':
    test_init()
    test_parse()
    test_eq()
    test_reduce()
    test_filter_key()
