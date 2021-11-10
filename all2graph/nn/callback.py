import os
import time

import torch
from sklearn.metrics import roc_auc_score

from ..utils import ks_score


class CallBack:
    def __init__(
            self, model, train_dataloader, valid_dataloader, target, save_dir=None, adj_factor=0.5,
            early_stopping_round=None, early_stopping_metric='adjust_auc'
    ):
        self.model = model
        self.save_dir = os.path.join(save_dir, time.asctime())
        os.mkdir(self.save_dir)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.target = target
        self.adj_factor = adj_factor
        self.early_stopping_round = early_stopping_round
        self.early_stopping_metric = early_stopping_metric
        self.history = []
        self.best_round = 0

    def __call__(self, ep, loss):
        if self.save_dir is not None:
            path = os.path.join(self.save_dir, '{}.torch.model'.format(str(ep)))
            print('save model at {}'.format(path))
            torch.save(self.model, path)

        train_pred, train_label = self.model.predict_dataloader(self.train_dataloader, 'eval train')
        valid_pred, valid_label = self.model.predict_dataloader(self.valid_dataloader, 'eval valid')

        metrics = {}
        for k in train_label:
            train_auc = roc_auc_score(train_label[k], train_pred[k])
            valid_auc = roc_auc_score(valid_label[k], valid_pred[k])
            adjust_auc = valid_auc - max((train_auc - valid_auc) * self.adj_factor, 0)

            train_ks = ks_score(train_label[k], train_pred[k])
            valid_ks = ks_score(valid_label[k], valid_pred[k])
            adjust_ks = valid_ks - max((train_ks - valid_ks) * self.adj_factor, 0)

            metrics[k] = {
                'train_auc': train_auc, 'valid_auc': valid_auc, 'adjust_auc': adjust_auc,
                'train_ks': train_ks, 'valid_ks': valid_ks, 'adjust_ks': adjust_ks
            }
            print('{}:'.format(k), ', '.join('{}={:.3}'.format(kk, vv) for kk, vv in metrics[k].items()))
        self.history.append(metrics)

        best_metric = self.history[self.best_round][self.target][self.early_stopping_metric]
        current_metric = metrics[self.target][self.early_stopping_metric]

        if best_metric < current_metric:
            self.best_round = ep

        if self.early_stopping_round is not None and ep - self.best_round > self.early_stopping_round:
            raise StopIteration