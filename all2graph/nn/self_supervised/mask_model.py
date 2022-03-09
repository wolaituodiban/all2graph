from typing import Dict

import torch

from ..framework import Framework
from ..utils import Module
from ...graph import Graph


class MaskModel(Module):
    def __init__(self, module: Framework, d_model, num_tokens, mask_token, p=0.01):
        super().__init__()
        self.module = module
        self.clf = torch.nn.Linear(d_model, num_tokens)
        self.reg = torch.nn.Linear(d_model, 1)
        self.mask_token = mask_token
        self.p = p
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.mean_square_error = torch.nn.MSELoss()
        self.to(self.module.device)

    @property
    def device(self):
        return self.module.device

    def reset_parameters(self):
        self.module.reset_parameters()
        self.clf.reset_parameters()
        self.reg.reset_parameters()

    def forward(self, graph: Graph) -> Dict[str, torch.Tensor]:
        graph = graph.to(self.device, non_blocking=True)
        with graph.local_scope():
            # mask原始数据
            mask = torch.rand(graph.num_nodes('value'), device=self.device) <= self.p
            token_label = graph.nodes['value'].data['token'][mask]
            num_label = graph.nodes['value'].data['number'][mask]
            graph.nodes['value'].data['token'] = torch.masked_fill(graph.nodes['value'].data['token'], mask,
                                                                   self.mask_token)
            graph.nodes['value'].data['number'] = torch.masked_fill(graph.nodes['value'].data['number'], mask, np.nan)

            # 提取特征
            graph = self.module(graph, details=True)
            value_feats = graph.nodes['value'].data['feats'][:, -1][mask]

            # 预测
            token_pred = self.clf(value_feats)
            num_pred = self.reg(value_feats).squeeze(-1)

            # 损失
            loss = self.cross_entropy(token_pred, token_label)
            acc = token_pred.argmax(dim=-1) == token_label
            acc = acc.sum() / acc.shape[0]
            mask = torch.bitwise_not(torch.isnan(num_label))
            if mask.sum() > 0:
                mse = self.mean_square_error(num_pred[mask], num_label[mask])
                loss += mse
            else:
                mse = 0

            return {'loss': loss, 'acc': acc, 'mse': mse}

    @staticmethod
    def loss(inputs, _=None):
        return inputs['loss']

    @staticmethod
    def acc(_, pred):
        return pred['acc'].mean()

    @staticmethod
    def mse(_, pred):
        return pred['mse'].mean()

    @property
    def metrics(self):
        return {'acc': self.acc, 'mse': self.mse}
