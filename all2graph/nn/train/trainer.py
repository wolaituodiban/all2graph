import os
from datetime import datetime as ddt
from typing import Dict, Callable, List

import torch
from torch.utils.data import DataLoader

from .early_stop import EarlyStop
from .history import History
from .metric import Metric
from ..utils import predict_dataloader
from ...utils import tqdm, json_round


class Trainer(torch.nn.Module):
    """trainer for training models
    """

    def __init__(
            self, module: torch.nn.Module, loss: torch.nn.Module, data: DataLoader,
            optimizer: torch.optim.Optimizer = None, scheduler=None, valid_data: List[DataLoader] = None,
            early_stop: EarlyStop = None, metrics: Dict[str, Callable] = None, callbacks: List[Callable] = None,
            valid_callbacks: List[Callable] = None, check_point=None):
        """

        Args:
            module:
            loss:
            optimizer:
            data:
            scheduler:
            valid_data:
            early_stop:
            metrics:
            callbacks:
            valid_callbacks:
            check_point: 保存路径
        """
        super().__init__()
        self.module = module
        self.loss = loss
        self.optimizer = optimizer or torch.optim.AdamW(self.module.parameters())
        self.scheduler = scheduler
        self.train_history = History(data)
        self.valid_history = [History(d) for d in valid_data or []]
        self.early_stop = early_stop
        self.metrics = [Metric(metric, name) for name, metric in (metrics or {}).items()]
        self.callbacks = callbacks or []
        self.valid_callbacks = valid_callbacks or []
        self.check_point = os.path.join(check_point, ddt.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
        os.mkdir(self.check_point)

        self._current_epoch = 0

    def extra_repr(self) -> str:
        output = ['optimizer={}'.format(self.optimizer)]
        if self.scheduler:
            output.append('scheduler={}'.format(self.scheduler))
        if self.early_stop:
            output.append('early_stop={}'.format(self.early_stop))
        if self.metrics:
            output.append('metrics=[\n{}\n]'.format(',\n'.join('  '+str(metric) for metric in self.metrics)))
        if self.check_point:
            output.append('check_point="{}"'.format(self.check_point))
        if self._current_epoch:
            output.append('current_epoch={}'.format(self._current_epoch))
        return ',\n'.join(output)

    def save(self):
        torch.save(self, os.path.join(self.check_point, '{}.all2graph.trainer'.format(self._current_epoch)))

    def train_one_epoch(self):
        self._current_epoch += 1
        with tqdm(list(range(len(self.train_history.loader))), desc='train epoch {}'.format(self._current_epoch)) as bar:
            self.module.train()
            for data, label in self.train_history.loader:
                # step fit
                self.optimizer.zero_grad()
                pred = self.module(data)
                loss = self.loss(pred, label)
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.train_history.log(pred=pred, loss=loss, label=label)
                bar.update()
            self.train_history.collate(epoch=self._current_epoch)
            bar.set_postfix({'loss': json_round(torch.mean(self.train_history.epochs[self._current_epoch].loss), 3)})

    def train(self, epochs=10):
        try:
            for _ in range(epochs):
                self.train_one_epoch()
                self.pred_valid()
                self.evaluate()
                if self.check_point:
                    self.save()
                if self.early_stop(self, None, self._current_epoch):
                    break
        except KeyboardInterrupt as e:
            raise e
        finally:
            self.save()

    def pred_valid(self):
        for i, valid_data in enumerate(self.valid_history):
            pred, label = predict_dataloader(self.module, valid_data.loader, desc='pred valid {}'.format(i))
            valid_data.add_epoch(self._current_epoch, pred=pred, label=label)

    def evaluate(self, epoch=None):
        epoch = epoch or self._current_epoch
        for metric in self.metrics:
            metric(trainer=self, history=self.train_history, epoch=epoch)
            print('train metrics: ', json_round(self.train_history.get_metric(epoch), 3))
            for i, valid_data in enumerate(self.valid_history):
                metric(trainer=self, history=valid_data, epoch=epoch)
                print('valid {} metrics: {}'.format(i, json_round(valid_data.get_metric(epoch), 3)))
