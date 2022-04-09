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


def test_body():
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

    d_model = 8
    in_feats = torch.randn((graph.num_nodes, d_model))
    module = ag.nn.Body(
        2,
        conv_layer=dgl.nn.pytorch.GATConv(d_model, d_model, 1),
        ff=ag.nn.FeedForward(d_model),
        seq_layer=torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=d_model, batch_first=True),
    )
    print(module)
    pred = module(graph.graph, in_feats, node2seq=graph.node2seq, seq2node=graph.seq2node(d_model),
                  seq_mask=torch.ones(graph.num_seqs, dtype=torch.bool))
    print(pred[-1].shape)
    pred[-1].sum().backward()
    for k, v in module.named_parameters():
        assert not torch.isnan(v.grad).any(), (k, v.grad)


if __name__ == '__main__':
    test_body()
