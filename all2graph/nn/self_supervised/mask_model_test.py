import platform
import os
import json
import torch.nn
import pandas as pd

import all2graph as ag
import numpy as np


if 'darwin' in platform.system().lower():
    os.environ['OMP_NUM_THREADS'] = '1'


def test_mask_model():
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
        json_col='json', time_col='crt_dte', time_format='%Y-%m-%d', targets=['m3_ovd_30'], local_foreign_key_types={'ord_no'})
    raw_graph = json_parser(df)
    meta_info = raw_graph.info()
    graph_parser = ag.GraphParser.from_data(meta_info)
    graph = graph_parser(raw_graph).add_self_loop()

    d_model = 8
    module = ag.nn.Framework(
        str_emb=torch.nn.Embedding(graph_parser.num_tokens, d_model),
        num_emb=ag.nn.NumEmb(d_model),
        bottle_neck=ag.nn.BottleNeck(d_model, num_inputs=3),
        key_emb=ag.nn.Body(d_model, num_heads=2, num_layers=2),
        body=ag.nn.Body(d_model, num_heads=2, num_layers=6),
        head=None
    )
    mask_model = ag.nn.MaskModel(module, 8, num_tokens=graph_parser.num_tokens, mask_token=graph_parser.mask_code)
    print(mask_model.cuda())
    pred = mask_model(graph)
    print(pred)
    loss = mask_model.loss(pred)
    loss.backward()
    for k, v in module.named_parameters():
        assert v.grad is not None, (k, v)
        assert not torch.isnan(v.grad).any(), (k, v.grad)
    pred = ag.nn.to_numpy(pred)
    for k, v in mask_model.metrics.items():
        print(k, v(None, pred))


if __name__ == '__main__':
    test_mask_model()
