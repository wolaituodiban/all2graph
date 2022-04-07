import platform
import os
import json

import dgl.nn.pytorch
import torch.nn
import pandas as pd

import all2graph as ag
import numpy as np


if 'darwin' in platform.system().lower():
    os.environ['OMP_NUM_THREADS'] = '1'


def test_block():
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
    meta_info = ag.MetaInfo.from_data(raw_graph)
    graph_parser = ag.GraphParser.from_data(meta_info)
    graph = graph_parser(raw_graph).add_self_loop()

    dim = 8
    in_feats = torch.randn((graph.num_nodes, dim))
    module = ag.nn.Block(dim, 2, conv_layer=dgl.nn.pytorch.GATConv(dim, dim, 1), dim_feedforward=dim)
    print(module)
    pred = module(graph.graph, in_feats, *graph.seq_masks())
    print(pred.shape)
    pred.sum().backward()
    for k, v in module.named_parameters():
        assert not torch.isnan(v.grad).any(), (k, v.grad)


if __name__ == '__main__':
    test_block()
