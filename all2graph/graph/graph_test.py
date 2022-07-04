import os
import string
import json

import numpy as np
import pandas as pd
import torch
import dgl

import all2graph as ag


if __name__ == '__main__':
    data = []
    for _ in ag.tqdm(range(100)):
        one_sample = []
        for _ in range(np.random.randint(1, 200)):
            item = {
                k: list(np.random.choice(list(string.ascii_letters)+list(string.digits), size=np.random.randint(1, 10)))
                for k in np.random.choice(list(string.ascii_letters), size=np.random.randint(1, 10))
            }
            one_sample.append(item)
        data.append(json.dumps(one_sample))
    df = pd.DataFrame({'data': data, 'target': np.random.choice([0, 1], size=len(data)), 'time': None})
    
    json_parser = ag.JsonParser(
        json_col='data', time_col='time', time_format='%Y-%m-%d', targets=['target'])

    meta_info = json_parser.analyse(df, chunksize=100, processes=0)
    graph_parser = ag.GraphParser.from_data(meta_info)
    parser_wrapper = ag.ParserWrapper(json_parser, graph_parser)

    graph = parser_wrapper(df)
    print(graph)
    graph1 = graph.add_self_loop()
    print(graph1)
    graph2 = graph.add_edges_by_seq(0, 0, add_self_loop=True)
    print(graph2)
    assert (graph1.edges()[0] == graph2.edges()[0]).all()
    assert (graph2.edges()[1] == graph2.edges()[1]).all()
    if torch.cuda.is_available():
        graph.pin_memory()
        graph.to('cuda', non_blocking=True)
