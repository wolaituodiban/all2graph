import distutils.dir_util
import os
import json
from datetime import datetime as ddt
from typing import Dict, Callable, List

import torch
from torch.utils.data import DataLoader

from .early_stop import EarlyStop
from .history import History, EpochBuffer
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
            valid_callbacks: List[Callable] = None, check_point=None, max_batch=None):
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
            max_batch: 当每个epoch训练的batch数量达到这个值时就会停止，可以用于防止batch太大时dataloader报mmap错误
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
        self.max_batch = max_batch

        if check_point:
            check_point += '.log'
            if not os.path.exists(check_point) or not os.path.isdir(check_point):
                os.mkdir(check_point)
            self.check_point = os.path.join(check_point, ddt.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
            os.mkdir(self.check_point)
        else:
            self.check_point = None

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
        self.module.eval()
        path = os.path.join(self.check_point, '{}.all2graph.trainer'.format(self._current_epoch))
        print('save at "{}"'.format(path))
        torch.save(self, path)

    def train_one_epoch(self, digits=3):
        self._current_epoch += 1
        with tqdm(list(range(len(self.train_history.loader))), desc='train epoch {}'.format(self._current_epoch)) as bar:
            self.module.train()
            buffer = EpochBuffer()
            for data, label in self.train_history.loader:
                # step fit
                self.optimizer.zero_grad()
                pred = self.module(data)
                loss = self.loss(pred, label)
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                buffer.log(pred=pred, loss=loss, label=label)
                bar.update()
                bar.set_postfix({'loss': json_round(buffer.mean_loss, digits)})
                if self.max_batch and buffer.batches >= self.max_batch:
                    break
            self.train_history.insert_buffer(epoch=self._current_epoch, buffer=buffer)
            bar.set_postfix({'loss': json_round(self.train_history.mean_loss(self._current_epoch), digits)})

    def train(self, epochs=10, digits=3, indent=None):
        try:
            for _ in range(epochs):
                self.train_one_epoch(digits=digits)
                self.pred_valid()
                self.evaluate(digits=digits, indent=indent)
                if self.check_point:
                    self.save()
                if self.early_stop(self, None, self._current_epoch):
                    break
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        finally:
            if self.check_point:
                self.save()

    def pred_valid(self):
        for i, valid_data in enumerate(self.valid_history):
            pred, label = predict_dataloader(self.module, valid_data.loader, desc='pred valid {}'.format(i))
            valid_data.add_epoch(self._current_epoch, pred=pred, label=label)

    def evaluate(self, epoch=None, digits=3, indent=None):
        epoch = epoch or self._current_epoch
        for metric in self.metrics:
            metric(trainer=self, history=self.train_history, epoch=epoch)
            for i, valid_data in enumerate(self.valid_history):
                metric(trainer=self, history=valid_data, epoch=epoch)
        print('train metrics: ', json.dumps(json_round(self.train_history.get_metric(epoch), digits), indent=indent))
        for i, valid_data in enumerate(self.valid_history):
            msg = json.dumps(json_round(valid_data.get_metric(epoch), digits), indent=indent)
            print('valid {} metrics: {}'.format(i, msg))
