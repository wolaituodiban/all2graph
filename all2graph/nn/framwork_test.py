import platform
import os
import json
import torch.nn
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
    meta_info = raw_graph.info()
    graph_parser = ag.GraphParser.from_data(meta_info)
    graph = graph_parser(raw_graph).add_self_loop()

    d_model = 8
    module = ag.nn.Framework(
        token_emb=torch.nn.Embedding(graph_parser.num_tokens, d_model),
        number_emb=ag.nn.NumEmb(d_model),
        bottle_neck=ag.nn.BottleNeck(d_model, num_inputs=3),
        key_body=ag.nn.GATBody(d_model, num_heads=2, num_layers=2),
        value_body=ag.nn.GATBody(d_model, num_heads=2, num_layers=6),
        readout=ag.nn.Readout(d_model)
    )
    print(module)
    pred = module(graph)
    pred['m3_ovd_30'].sum().backward()
    for k, v in module.named_parameters():
        assert not torch.isnan(v.grad).any(), (k, v.grad)


if __name__ == '__main__':
    test_framework()
