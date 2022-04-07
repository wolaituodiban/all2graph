import platform
import os
import json
import torch.nn
import dgl.nn.pytorch
import pandas as pd

import all2graph as ag
import numpy as np


if 'darwin' in platform.system().lower():
    os.environ['OMP_NUM_THREADS'] = '1'


def get_metric(x):
    return x['auc']['m3_ovd_30']


def test_framework():
    data = [
               {
                   'ord_no': 'CH202007281033864',
                   'bsy_typ': 'CASH',
               },
               {
                   'ord_no': 'CH202007281033864',
                   'stg_no': '1',
               },
           ]
    df = pd.DataFrame({'json': [json.dumps(data)] * 2, 'crt_dte': '2020-10-09'})
    df['m3_ovd_30'] = np.random.choice([0, 1], size=df.shape[0])

    json_parser = ag.JsonParser(
        json_col='json', time_col='crt_dte', time_format='%Y-%m-%d', targets=['m3_ovd_30'], lid_keys={'ord_no'})
    raw_graph = json_parser(df)
    meta_info = ag.MetaInfo.from_data(raw_graph)
    graph_parser = ag.GraphParser.from_data(meta_info)
    graph = graph_parser(raw_graph).add_self_loop()

    d_model = 8
    module = ag.nn.Framework(
        key_emb=torch.nn.RNN(d_model, d_model, 1),
        str_emb=torch.nn.Embedding(graph_parser.num_tokens, d_model),
        num_emb=ag.nn.NumEmb(d_model),
        bottle_neck=ag.nn.BottleNeck(d_model),
        body=ag.nn.Body(
            num_layers=6,
            conv_layer=dgl.nn.pytorch.GATConv(d_model, d_model, 1, residual=True),
            seq_layer=torch.nn.RNN(d_model, d_model, 1),
        ),
        head=ag.nn.Head(d_model)
    )
    if torch.cuda.is_available():
        module.cuda()
    print(module)
    pred = module(graph)
    pred['m3_ovd_30'].sum().backward()
    for k, v in module.named_parameters():
        assert not torch.isnan(v.grad).any(), (k, v.grad)


if __name__ == '__main__':
    test_framework()
