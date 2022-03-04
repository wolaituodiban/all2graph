import platform
import os
import json
import torch.nn
import pandas as pd

import all2graph as ag
import numpy as np
from sklearn.metrics import roc_auc_score


if 'darwin' in platform.system().lower():
    os.environ['OMP_NUM_THREADS'] = '1'


def get_metric(x):
    return x['auc']['m3_ovd_30']


def test_gat_model():
    data = [
               {
                   'ord_no': 'CH202007281033864',
                   'bsy_typ': 'CASH',
               },
               {
                   'ord_no': 'CH202007281033864',
                   'stg_no': '1',
               },
           ] * 10
    df = pd.DataFrame({'json': [json.dumps(data)], 'crt_dte': '2020-10-09'})
    df = pd.concat([df] * 1000)
    df['m3_ovd_30'] = np.random.choice([0, 1], size=df.shape[0])

    json_parser = ag.JsonParser(
        json_col='json', time_col='crt_dte', time_format='%Y-%m-%d', targets=['m3_ovd_30'], lid_keys={'ord_no'})
    gat_model = ag.nn.GATModel(
        data_parser=json_parser,
        check_point='temp',
        d_model=8,
        num_key_layers=2,
        num_value_layers=6,
        num_heads=2,
        out_feats=1,
        dropout=0.5,
        activation='prelu',
        norm_first=False,
        meta_info_configs=dict(num_bins=100),
        graph_parser_configs=dict(min_df=0.01),
        post_parser=ag.PostParser(degree=-1, r_degree=-1),
        residual=True
    )

    gat_model.fit(
        train_data=df,
        chunksize=100,
        batch_size=16,
        processes=0,
        loss=ag.nn.DictLoss(torch.nn.BCEWithLogitsLoss()),
        epoches=2,
        valid_data=[df],
        optimizer_cls=torch.optim.Adam,
        optimizer_kwds=dict(lr=0.0001),
        metrics={'auc': ag.Metric(roc_auc_score, label_first=True)},
        early_stop=ag.nn.EarlyStop(5, higher=True, fn=get_metric)
    )


if __name__ == '__main__':
    test_gat_model()
