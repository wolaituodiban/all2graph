import os
import string
import json

import numpy as np
import pandas as pd
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
    graph.pin_memory()
    graph.to('cuda', non_blocking=True)
