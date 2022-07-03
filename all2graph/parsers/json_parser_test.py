import pandas as pd
import matplotlib.pyplot as plt
import all2graph as ag


def test_targets():
    json_parser = ag.JsonParser(json_col='json', time_col='crt_dte', time_format='%y-%m-%d', targets=['t7', 't1'])
    graph = json_parser(df)
    graph._assert()
    fig, ax = plt.subplots(figsize=(16, 8))
    graph.draw(ax=ax)
    plt.title('test_d_targets')
    plt.show()


def test_targets2():
    json_parser = ag.JsonParser(
        json_col='json', time_col='crt_dte', time_format='%y-%m-%d', targets={'a_b': ('a', 'b'), 'c_d': ('c', 'd')})
    graph = json_parser(df)
    graph._assert()
    fig, ax = plt.subplots(figsize=(16, 8))
    graph.draw(ax=ax)
    plt.title('test_d_targets2')
    plt.show()


def test_d_degree():
    json_parser = ag.JsonParser(json_col='json', time_col='crt_dte', time_format='%y-%m-%d', dict_degree=2)
    graph = json_parser(df)
    graph._assert()
    fig, ax = plt.subplots(figsize=(16, 8))
    graph.draw(ax=ax)
    plt.title('test_d_degree')
    plt.show()


def test_d_inner_edge():
    json_parser = ag.JsonParser(json_col='json', time_col='crt_dte', time_format='%y-%m-%d', dense_dict=True)
    graph = json_parser(df)
    graph._assert()
    fig, ax = plt.subplots(figsize=(16, 8))
    graph.draw(ax=ax)
    plt.title('test_d_inner_edge')
    plt.show()


def test_l_degree():
    json_parser = ag.JsonParser(json_col='json', time_col='crt_dte', time_format='%y-%m-%d', list_degree=0)
    graph = json_parser(df)
    graph._assert()
    fig, ax = plt.subplots(figsize=(16, 8))
    graph.draw(ax=ax)
    plt.title('test_l_degree')
    plt.show()


def test_lid_keys():
    json_parser = ag.JsonParser(json_col='json', time_col='crt_dte', time_format='%y-%m-%d', local_foreign_key_types={'ord_no'})
    graph = json_parser(pd.concat([df]))
    graph._assert()
    fig, ax = plt.subplots(figsize=(16, 8))
    graph.draw(ax=ax)
    plt.title('test_lid_keys')
    plt.show()


def test_gid_keys():
    json_parser = ag.JsonParser(json_col='json', time_col='crt_dte', time_format='%y-%m-%d', global_foreign_key_types={'ord_no'})
    graph = json_parser(pd.concat([df] * 2))
    graph._assert()
    fig, ax = plt.subplots(figsize=(16, 8))
    graph.draw(ax=ax, pos=None)
    plt.title('test_gid_keys')
    plt.show()


def test_analyse():
    json_parser = ag.JsonParser(json_col='json', time_col='crt_dte', time_format='%y-%m-%d', targets={'label'}, global_foreign_key_types={'ord_no'})
    print(json_parser)
    meta_info1 = json_parser.analyse(pd.concat([df] * 10000), processes=0)
    meta_info2 = json_parser.analyse(pd.concat([df] * 10000))
    assert meta_info1 == meta_info2


if __name__ == '__main__':
    data = [
        {
            'ord_no': 'CH202007281033864',
            'bst_typ': 'CASH',
        },
        {
            'ord_no': 'CH202007281033864',
            'stg_no': '1',
        },
    ]
    df = pd.DataFrame(
        {
            'json': [data],
            'crt_dte': '2020-10-09'
        }
    )

    test_targets()
    test_targets2()
    test_d_degree()
    test_d_inner_edge()
    test_l_degree()
    test_lid_keys()
    test_gid_keys()

    df = pd.DataFrame(
        {
            'json': [data * 100],
            'crt_dte': '2020-10-09'
        }
    )

    test_analyse()
