import os

import numpy as np
import pandas as pd
import string
import json
import all2graph as ag
import torch
import dgl.nn.pytorch


if __name__ == '__main__':
    dir_path = os.path.join(os.path.dirname(__file__), ag.__version__)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    data_path = os.path.join(dir_path, 'test_data.csv.zip')
    parser_wrapper_path = os.path.join(dir_path, 'parser_wrapper.json')
    framework_path = os.path.join(dir_path, 'framework.th')
    data = []
    for _ in range(100):
        one_sample = []
        for _ in range(np.random.randint(1, 200)):
            item = {
                k: list(
                    np.random.choice(list(string.ascii_letters) + list(string.digits), size=np.random.randint(1, 10)))
                for k in np.random.choice(list(string.ascii_letters), size=np.random.randint(1, 10))
            }
            one_sample.append(item)
        data.append(json.dumps(one_sample))
    df = pd.DataFrame({'data': data, 'target': np.random.choice([0, 1], size=len(data)), 'time': None})

    json_parser = ag.JsonParser(
        json_col='data', time_col='time', time_format='%Y-%m-%d', targets=['target'])
    meta_info = ag.MetaInfo.from_data(json_parser(df))
    graph_parser = ag.GraphParser.from_data(meta_info)
    parser_wrapper = ag.ParserWrapper(json_parser, graph_parser)

    d_model = 8
    num_layers = 6
    body = ag.nn.Body(
        num_layers,
        conv_layer=dgl.nn.pytorch.GATConv(d_model, d_model, 1, residual=True),
        ff=ag.nn.FeedForward(d_model, pre=torch.nn.BatchNorm1d(d_model)),
    )
    framework = ag.nn.Framework(
        key_emb=torch.nn.LSTM(d_model, d_model // 2, 2, bidirectional=True, batch_first=True),
        str_emb=torch.nn.Embedding(graph_parser.num_tokens, d_model),
        num_emb=ag.nn.NumEmb(d_model),
        bottle_neck=ag.nn.BottleNeck(d_model),
        body=body,
        head=ag.nn.Head((num_layers + 1) * d_model),
        seq_degree=(10, 10)
    )

    model = ag.nn.Model(parser=parser_wrapper, module=framework)
    print(model(df))
    model.predict(df, processes=0, drop_data_cols=False).to_csv(data_path, index=False)
    with open(parser_wrapper_path, 'w') as file:
        json.dump(parser_wrapper.to_json(), file)
    torch.save(model.module, framework_path)
