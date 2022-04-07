from typing import Dict

import torch
import numpy as np

from ..bottle_neck import BottleNeck
from ..feedforward import FeedForward
from ..framework import Framework
from ..utils import Module
from ...graph import Graph
from ...globals import VALUE, STRING, NUMBER


class SGP(Module):
    def __init__(self, d_model, dropout=0., activation='relu', norm_first=True):
        super().__init__()
        self.bottle_neck = BottleNeck(
            d_model, num_inputs=2, dropout=dropout, activation=activation, norm_first=norm_first)
        self.predictor = torch.nn.Linear(d_model, 1)

    @property
    def device(self):
        return self.bottle_neck.device

    def reset_parameters(self):
        self.bottle_neck.reset_parameters()
        self.predictor.reset_parameters()

    def forward(self, x, y):
        assert x.shape[0] >= 2
        # 由于当样本数量为奇数时，倒排之后的中间那个依然是样本，所以要把样本变成偶数个
        if x.shape[0] % 2 != 0:
            x = x[1:]
            y = y[1:]
        pos_pred = self.predictor(self.bottle_neck(torch.cat([x, y], dim=0), torch.cat([y, x], dim=0)))
        # 倒排构造负样本
        y = y[torch.arange(y.shape[0]-1, -1, -1)]
        neg_pred = self.predictor(self.bottle_neck(torch.cat([x, y], dim=0), torch.cat([y, x], dim=0)))
        return pos_pred.squeeze(-1), neg_pred.squeeze(-1)


class MaskModel(Module):
    def __init__(self, module: Framework, d_model, num_tokens, mask_token, sgp=True,
                 p=0.1, activation='relu', norm_first=True):
        super().__init__()
        self.module = module
        self.clf = FeedForward(
            d_model, out_feats=num_tokens, dropout=p, activation=activation, norm_first=norm_first, residual=False
        ).to(module.device)
        self.reg = FeedForward(
            d_model, out_feats=1, dropout=p, activation=activation, norm_first=norm_first, residual=False
        ).to(module.device)

        self.mask_token = mask_token
        self.p = p
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.mean_square_error = torch.nn.MSELoss()
        if sgp:  # subgraph prediction
            self.sgp = SGP(d_model, dropout=p, activation=activation, norm_first=norm_first)
            self.bce_with_logits = torch.nn.BCEWithLogitsLoss()

    @property
    def device(self):
        return self.module.device

    def reset_parameters(self):
        self.module.reset_parameters()
        self.clf.reset_parameters()
        self.reg.reset_parameters()
        if hasattr(self, 'sgp'):
            self.sgp.reset_parameters()

    def forward(self, graph: Graph) -> Dict[str, torch.Tensor]:
        mask1 = torch.rand(graph.num_nodes(VALUE), device=self.device) <= self.p
        if hasattr(self, 'sgp'):
            root_ids, _ = graph.edges(etype=(VALUE, EDGE, graph.readout_types[0]))
            mask2 = torch.bitwise_not(mask1)
            mask1[root_ids] = True
            mask2[root_ids] = True
            # 因为subgraph无法在gpu上完成，所以这段操作要在这完成
            subgraph1 = graph.value_subgraph(mask1, store_ids=False).to(self.device, non_blocking=True)
            subgraph2 = graph.value_subgraph(mask2, store_ids=False).to(self.device, non_blocking=True)

            emb1 = self.module(subgraph1, ret_graph_emb=True)
            emb2 = self.module(subgraph2, ret_graph_emb=True)
        else:
            emb1, emb2 = None, None

        graph = graph.to(self.device, non_blocking=True)
        with graph.local_scope():
            # mask原始数据
            token_label = graph.nodes[VALUE].data[STRING][mask1]
            num_label = graph.nodes[VALUE].data[NUMBER][mask1]
            graph.nodes[VALUE].data[STRING] = torch.masked_fill(graph.nodes[VALUE].data[STRING], mask1, self.mask_token)
            graph.nodes[VALUE].data[NUMBER] = torch.masked_fill(graph.nodes[VALUE].data[NUMBER], mask1, np.nan)

            # 提取特征
            graph = self.module(graph, details=True)
            value_feats = graph.nodes[VALUE].data['feats'][:, -1][mask1]

            # 预测
            token_pred = self.clf(value_feats)
            num_pred = self.reg(value_feats).squeeze(-1)

            # 损失
            output = dict()
            output['ce'] = self.cross_entropy(token_pred, token_label)
            acc = token_pred.argmax(dim=-1) == token_label
            output['acc'] = acc.sum() / acc.shape[0]
            mask1 = torch.bitwise_not(torch.isnan(num_label))
            if mask1.sum() > 0:
                output['mse'] = self.mean_square_error(num_pred[mask1], num_label[mask1])
            else:
                output['mse'] = torch.tensor(0)

            if emb1 is not None:
                emb = graph.nodes[graph.readout_types[0]].data['value_feats']
                output['sgp_pos_01'], output['sgp_neg_01'] = self.sgp(emb, emb1)
                output['sgp_pos_02'], output['sgp_neg_02'] = self.sgp(emb, emb2)
                output['sgp_pos_12'], output['sgp_neg_12'] = self.sgp(emb1, emb2)

            return output

    def loss(self, inputs, _=None):
        loss = inputs['ce'] + inputs['mse']
        if hasattr(self, 'sgp'):
            num_pos = inputs['sgp_pos_01'].shape[0] + inputs['sgp_pos_02'].shape[0] + inputs['sgp_pos_12'].shape[0]
            num_neg = inputs['sgp_neg_01'].shape[0] + inputs['sgp_neg_02'].shape[0] + inputs['sgp_neg_12'].shape[0]
            pred = torch.cat([
                inputs['sgp_pos_01'], inputs['sgp_pos_02'], inputs['sgp_pos_12'],
                inputs['sgp_neg_01'], inputs['sgp_neg_02'], inputs['sgp_neg_12'],
            ], dim=0)
            targets = torch.ones(num_pos + num_neg, dtype=torch.float32, device=loss.device)
            targets[num_pos:] = 0
            loss = loss + self.bce_with_logits(pred, targets)
        return loss

    @staticmethod
    def acc(_, pred):
        return pred['acc'].mean()

    @staticmethod
    def mse(_, pred):
        return pred['mse'].mean()

    @staticmethod
    def auc(_, pred):
        from sklearn.metrics import roc_auc_score
        num_pos = pred['sgp_pos_01'].shape[0] + pred['sgp_pos_02'].shape[0] + pred['sgp_pos_12'].shape[0]
        num_neg = pred['sgp_neg_01'].shape[0] + pred['sgp_neg_02'].shape[0] + pred['sgp_neg_12'].shape[0]
        pred = np.concatenate([
            pred['sgp_pos_01'], pred['sgp_pos_02'], pred['sgp_pos_12'],
            pred['sgp_neg_01'], pred['sgp_neg_02'], pred['sgp_neg_12'],
        ], axis=0)
        targets = np.ones(num_pos + num_neg)
        targets[num_pos:] = 0
        return roc_auc_score(targets, pred)

    @staticmethod
    def auc01(_, pred):
        from sklearn.metrics import roc_auc_score
        num_pos = pred['sgp_pos_01'].shape[0]
        num_neg = pred['sgp_neg_01'].shape[0]
        pred = np.concatenate([pred['sgp_pos_01'], pred['sgp_neg_01']], axis=0)
        targets = np.ones(num_pos + num_neg)
        targets[num_pos:] = 0
        return roc_auc_score(targets, pred)

    @staticmethod
    def auc02(_, pred):
        from sklearn.metrics import roc_auc_score
        num_pos = pred['sgp_pos_02'].shape[0]
        num_neg = pred['sgp_neg_02'].shape[0]
        pred = np.concatenate([pred['sgp_pos_02'], pred['sgp_neg_02']], axis=0)
        targets = np.ones(num_pos + num_neg)
        targets[num_pos:] = 0
        return roc_auc_score(targets, pred)

    @staticmethod
    def auc12(_, pred):
        from sklearn.metrics import roc_auc_score
        num_pos = pred['sgp_pos_12'].shape[0]
        num_neg = pred['sgp_neg_12'].shape[0]
        pred = np.concatenate([pred['sgp_pos_12'], pred['sgp_neg_12']], axis=0)
        targets = np.ones(num_pos + num_neg)
        targets[num_pos:] = 0
        return roc_auc_score(targets, pred)

    @property
    def metrics(self):
        return {'acc': self.acc, 'mse': self.mse, 'auc': self.auc,
                'auc01': self.auc01, 'auc02': self.auc02, 'auc12': self.auc12}
