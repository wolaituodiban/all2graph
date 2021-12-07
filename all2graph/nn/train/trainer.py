import os
import json
import sys
import traceback
from datetime import datetime as ddt
from typing import Dict, Callable, List, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .early_stop import EarlyStop
from .history import History, EpochBuffer
from .metric import Metric
from ..utils import predict_dataloader, Predictor, MyModule
from ...utils import tqdm, json_round
from ...version import __version__


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
            check_point = '.'.join([check_point, __version__])
            if not os.path.exists(check_point) or not os.path.isdir(check_point):
                os.mkdir(check_point)
            self.check_point = os.path.join(check_point, ddt.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
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

    def save(self):
        self.module.eval()
        path = os.path.join(self.check_point, '{}.all2graph.trainer'.format(self._current_epoch))
        print('save at "{}"'.format(path))
        torch.save(self, path)

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

    def fit(self, epochs=10, digits=3, indent=None):
        try:
            for _ in range(epochs):
                self.fit_one_epoch(digits=digits)
                self.pred_valid()
                self.evaluate(digits=digits, indent=indent)
                if self.check_point:
                    self.save()
                if self.early_stop is not None and self.early_stop(self, None, self._current_epoch):
                    break
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        except:
            self.error_msg = traceback.format_exc()
            print(self.error_msg)
        finally:
            if self.check_point:
                self.save()

    def pred_valid(self):
        for i, valid_data in enumerate(self.valid_history):
            pred, label = predict_dataloader(
                self.module, valid_data.loader, desc='epoch {} val {}'.format(self._current_epoch, i)
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

    def reset_dataloader(self, *args, valid_id=None, **kwargs):
        """
        重新设置dataloader的参数
        Args:
            *args: DataLoader的参数
            valid_id: 如果None，那么重置train dataloder，否则重置对应的valid dataloader
            **kwargs: DataLoader的参数

        Returns:

        """
        if valid_id is None:
            self.train_history.reset_dataloader(*args, **kwargs)
        else:
            self.valid_history[valid_id].reset_dataloader(*args, **kwargs)

    def build_predictor(self, valid_id=None) -> Union[Predictor, None]:
        """
        生成一个predictor
        Args:
            valid_id: 如果None，那么使用train dataloader的data parser，否则使用对应valid dataloader的data parser

        Returns:

        """
        if valid_id is None:
            data_parser = self.train_history.get_data_parser()
        else:
            data_parser = self.valid_history[0].get_data_parser()

        if data_parser is None:
            print('failed to build predictor', file=sys.stderr)
            return
        if not isinstance(self.module, MyModule):
            print('module is not a all2graph module, can build predictor', file=sys.stderr)
            return

        return Predictor(data_parser=data_parser, module=self.module)

    def predict(self, src: Union[str, List[str]], valid_id=None, **kwargs) -> pd.DataFrame:
        """
        自动将预测结果保存在check point目录下
        Args:
            src: file path or list of file path
            valid_id: 默认选择train dataset的data parser
            kwargs: 传递给Predictor.predict的参数

        Returns:

        """
        if isinstance(src, list):
            return pd.concat(map(self.predict, src))

        dst = os.path.join(self.check_point, str(self._current_epoch))
        if not os.path.exists(dst):
            os.mkdir(dst)
        dst = os.path.join(dst, 'pred_{}'.format(os.path.split(src)[-1]))
        print("save prediction at '{}'".format(dst))

        predictor = self.build_predictor(valid_id=valid_id)
        if predictor is None:
            return pd.DataFrame()
        output = predictor.predict(src, **kwargs)
        output.to_csv(dst)
        return output
