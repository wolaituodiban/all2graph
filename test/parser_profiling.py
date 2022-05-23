import cProfile
import os
import all2graph as ag
import torch
import pandas as pd


if __name__ == '__main__':
    data_filename = '0.4.0.f7ff01da18e005eccd11753247c5c32cb6d76ef9/test_data.csv.zip'
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_filename)
    print(data_path)
    data_df = pd.read_csv(data_path)

    json_parser = ag.JsonParser(json_col='data', time_col='time', dense_dict=True, targets=['target'])
    # baseline 1.067
    # cython 0.377
    # cProfile.run("json_parser(data_df).seq_info()", sort='time')

    raw_graph = json_parser(data_df)
    meta_info = ag.MetaInfo.from_data(raw_graph)
    graph_parser = ag.GraphParser.from_data(meta_info)
    parser_wrapper = ag.ParserWrapper(data_parser=json_parser, graph_parser=graph_parser)
    # baseline 1.459
    # cython 0.623
    # torch.save(parser_wrapper, 'parser_wrapper.th')
    # parser_wrapper = torch.load('parser_wrapper.th')
    # print(parser_wrapper(data_df))
    cProfile.run("parser_wrapper(data_df)", sort='time')
