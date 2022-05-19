import all2graph as ag
import pandas as pd
import json


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
    json_parser = ag.JsonParser(json_col='json', time_col='crt_dte', time_format='%y-%m-%d', global_foreign_key_types={'ord_no'})
    meta_info = json_parser.analyse(df, processes=0)
    graph_parser = ag.GraphParser.from_data(meta_info)

    parser_wrapper = ag.ParserWrapper(
        data_parser=json_parser,
        graph_parser=graph_parser
    )
    graphs = parser_wrapper(df)
    print(graphs)
    print(json.dumps(parser_wrapper.to_json()))

    parser_wrapper = ag.ParserWrapper(
        data_parser={'a': json_parser, 'b': json_parser},
        graph_parser={'c': graph_parser, 'd': (graph_parser, ['a'])},
    )
    print(parser_wrapper)
    graphs, df2 = parser_wrapper.generate(df)
    print(graphs)
    print(df2)

    parser_json = parser_wrapper.to_json()
    # print(json.dumps(parser_json, indent=1))
    parser_wrapper = ag.ParserWrapper.from_json(parser_json)
    print(parser_wrapper)


if __name__ == '__main__':
    test_parse()
