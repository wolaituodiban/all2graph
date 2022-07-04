import copy
import os
import json
import traceback
from datetime import datetime as ddt
from typing import Dict, Callable, List, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .early_stop import EarlyStop
from .history import History, EpochBuffer
from .metric import Metric
from ..utils import predict_dataloader, predict_csv
from ...utils import tqdm, json_round
from ...version import __version__


class Trainer(torch.nn.Module):
    """trainer for training models
    """

    def __init__(
            self, module: torch.nn.Module,  data: DataLoader, loss: torch.nn.Module = None,
            optimizer: torch.optim.Optimizer = None, scheduler=None, valid_data: List[DataLoader] = None,
            early_stop: EarlyStop = None, metrics: Dict[str, Callable] = None, callbacks: List[Callable] = None,
            valid_callbacks: List[Callable] = None, check_point=None, max_batch=None, max_history=1,
            save_loader=True):
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
            max_batch: 当每个epoch训练的batch数量达到这个值时就会停止, 可以用于防止batch太大时dataloader报mmap错误
            max_history: 保留历史的epoch数量
            save_loader: 是否保存data loader
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
        self.max_history = max_history
        self.save_loader = save_loader

        if check_point:
            check_point = '.'.join([check_point, __version__])
            if not os.path.exists(check_point) or not os.path.isdir(check_point):
                os.mkdir(check_point)
            self.check_point = os.path.join(check_point, ddt.now().strftime('%Y%m%d%H%M%S.%f'))
            os.mkdir(self.check_point)
        else:
            self.check_point = None

        self._current_epoch = 0

        self.error_msg = None

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

    @property
    def current_epoch(self):
        return copy.deepcopy(self._current_epoch)

    @property
    def path(self):
        return os.path.join(self.check_point, '{}.all2graph.trainer'.format(self._current_epoch))

    @property
    def best_metric(self):
        return self.early_stop.best_metric

    @property
    def sign(self):
        return self.early_stop.sign

    def save(self):
        self.module.eval()
        print('save at "{}"'.format(self.path))
        if self.save_loader:
            torch.save(self, self.path)
        else:
            train_loader = self.get_data_loader()
            self.train_history.loader = None
            valid_loaders = []
            for valid_history in self.valid_history:
                valid_loaders.append(valid_history.loader)
                valid_history.loader = None
            torch.save(self, self.path)
            self.set_data_loader(train_loader)
            for i, loader in enumerate(valid_loaders):
                self.set_data_loader(loader, valid_id=i)

    def fit_one_epoch(self, digits=3):
        self._current_epoch += 1
        with tqdm(
                list(range(len(self.train_history.loader))),
                desc='epoch {} train'.format(self._current_epoch)
        ) as bar:
            self.module.train()
            buffer = EpochBuffer()
            for data, label in self.train_history.loader:
                # step fit
                self.optimizer.zero_grad()
                pred = self.module(data)
                if self.loss:
                    loss = self.loss(pred, label)
                else:
                    loss = pred
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

    def fit(self, epochs=10, digits=3, indent=None):
        try:
            for _ in range(epochs):
                self.fit_one_epoch(digits=digits)
                self.pred_valid()
                self.evaluate(digits=digits, indent=indent)
                if self.max_history is not None:
                    self.delete_history()
                if self.check_point:
                    self.save()
                if self.early_stop is not None and self.early_stop(self, None, self._current_epoch):
                    break
        except (KeyboardInterrupt, SystemExit):
            print('KeyboardInterrupt')
        except:
            self.error_msg = traceback.format_exc()
            print(self.error_msg)
        finally:
            if self.check_point:
                self.save()
            if self.early_stop is not None:
                self._current_epoch = self.early_stop._best_epoch
                if self.check_point is not None:
                    self.module = torch.load(self.path).module

    def pred_valid(self):
        for i, valid_data in enumerate(self.valid_history):
            pred, label = predict_dataloader(
                self.module, valid_data.loader, desc='epoch {} val {}'.format(self._current_epoch, i),
                max_batch=self.max_batch
            )
            valid_data.add_epoch(self._current_epoch, pred=pred, label=label)

    def evaluate(self, epoch=None, digits=3, indent=None):
        epoch = epoch or self._current_epoch
        for metric in self.metrics:
            metric(trainer=self, history=self.train_history, epoch=epoch)
            for i, valid_data in enumerate(self.valid_history):
                metric(trainer=self, history=valid_data, epoch=epoch)
        print(
            'epoch {} train metrics:'.format(epoch),
            json.dumps(json_round(self.train_history.get_metric(epoch), digits), indent=indent)
        )
        for i, valid_data in enumerate(self.valid_history):
            msg = json.dumps(json_round(valid_data.get_metric(epoch), digits), indent=indent)
            print('epoch {} val {} metrics: {}'.format(epoch, i, msg))

    def get_history(self, valid_id=None) -> History:
        if valid_id is None:
            return self.train_history
        else:
            return self.valid_history[valid_id]

    def get_dataset(self, valid_id=None):
        history = self.get_history(valid_id=valid_id)
        if history is not None:
            return history.dataset

    def get_data_loader(self, valid_id=None) -> DataLoader:
        history = self.get_history(valid_id)
        if history is not None:
            return history.loader

    def get_parser(self, valid_id=None):
        return self.get_history(valid_id=valid_id).parser

    def get_pred(self, epoch=None, valid_id=None):
        epoch = epoch or self._current_epoch
        return self.get_history(valid_id=valid_id).get_pred(epoch=epoch)

    def get_label(self, epoch=None, valid_id=None):
        epoch = epoch or self._current_epoch
        return self.get_history(valid_id=valid_id).get_label(epoch=epoch)

    def get_metric(self, epoch=None, key=None, valid_id=None):
        epoch = epoch or self._current_epoch
        return self.get_history(valid_id=valid_id).get_metric(epoch=epoch, key=key)

    def get_mean_loss(self, epoch=None, valid_id=None):
        epoch = epoch or self._current_epoch
        return self.get_history(valid_id=valid_id).mean_loss(epoch=epoch)

    def get_loss(self, epoch=None, valid_id=None):
        epoch = epoch or self._current_epoch
        return self.get_history(valid_id=valid_id).get_loss(epoch)

    def set_data_loader(self, loader: DataLoader, valid_id=None):
        self.get_history(valid_id=valid_id).loader = loader

    def add_valid_data(self, loader: DataLoader):
        self.valid_history.append(History(loader))

    def predict(self, src: Union[str, List[str]], valid_id=None, data_parser=None, **kwargs) -> pd.DataFrame:
        """
        自动将预测结果保存在check point目录下
        Args:
            src: file path or list of file path
            valid_id: 如果None, 那么使用train dataloader的data parser, 否则使用对应valid dataloader的data parser
            data_parser: 如果提供了自定义的data parser, 那么valid_id将不生效
            kwargs: 传递给Predictor.predict的参数

        Returns:

        """
        if isinstance(src, list):
            return pd.concat(self.predict(path, valid_id=valid_id, data_parser=data_parser, **kwargs) for path in src)

        dst = os.path.join(self.check_point, str(self._current_epoch))
        if not os.path.exists(dst):
            os.mkdir(dst)
        dst = os.path.join(dst, 'pred_{}.zip'.format(os.path.split(src)[-1]))
        output = predict_csv(self.get_parser(valid_id=valid_id), self.module, src, **kwargs)
        print("save prediction at '{}'".format(dst))
        output.to_csv(dst)
        return output

    def delete_history(self, epochs=None):
        """

        Args:
            epochs: 保留多少个epoch的history, 默认self.max_history

        Returns:

        """
        epochs = epochs or self.max_history
        self.train_history.delete_history(epochs=epochs)
        for valid_history in self.valid_history:
            valid_history.delete_history(epochs=epochs)
