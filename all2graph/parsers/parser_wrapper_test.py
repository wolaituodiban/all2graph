import all2graph as ag
import pandas as pd


def test_parse():
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
    json_parser = ag.JsonParser(json_col='json', time_col='crt_dte', time_format='%y-%m-%d', gid_keys={'ord_no'})
    meta_info = json_parser.analyse(df, processes=0)
    graph_parser = ag.GraphParser.from_data(meta_info)

    parser_wrapper = ag.ParserWrapper(
        data_parser=json_parser,
        graph_parser=graph_parser
    )
    graphs = parser_wrapper(df, return_df=False)
    print(graphs)

    parser_wrapper = ag.ParserWrapper(
        data_parser={'a': json_parser, 'b': json_parser},
        graph_parser={'c': graph_parser, 'd': (graph_parser, ['a'])},
    )
    print(parser_wrapper)
    graphs, df2 = parser_wrapper(df, return_df=True)
    print(graphs)


if __name__ == '__main__':
    test_parse()
