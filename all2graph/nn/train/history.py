import sys
from typing import Dict, Union

import torch
from torch.utils.data import DataLoader
from ..utils import detach, default_collate
from ...data import CSVDataset
from ...parsers import DataParser


class EpochBuffer:
    def __init__(self):
        self.pred = []
        self.label = []
        self.loss = []
        self.batches = 0
        self.mean_loss = 0

    @torch.no_grad()
    def log(self, pred, label, loss):
        self.pred.append(detach(pred))
        self.label.append(detach(label))
        self.batches += 1
        self.loss.append(detach(loss))
        self.mean_loss += (detach(loss) - self.mean_loss) / self.batches


class Epoch:
    def __init__(self, pred, label, loss=None):
        self.pred = pred
        self.label = label
        self.loss = loss
        self.metric = {}

    @classmethod
    @torch.no_grad()
    def from_buffer(cls, buffer: EpochBuffer):
        pred = default_collate(buffer.pred)
        label = default_collate(buffer.label)
        loss = default_collate(buffer.loss)
        return cls(pred=pred, label=label, loss=loss)


class History:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.epochs: Dict[int, Epoch] = {}

    @property
    def num_epochs(self):
        return len(self.epochs)

    def pop(self):
        min_epoch = min(list(self.epochs))
        output = self.epochs[min_epoch]
        del self.epochs[min_epoch]
        return output

    @property
    def last(self) -> Epoch:
        max_epoch = max(list(self.epochs))
        return self.epochs[max_epoch]

    def get_label(self, epoch):
        if epoch in self.epochs:
            return self.epochs[epoch].label

    def get_pred(self, epoch):
        if epoch in self.epochs:
            return self.epochs[epoch].pred

    def get_metric(self, epoch, key=None):
        if epoch in self.epochs:
            metric = self.epochs[epoch].metric
            if key:
                return metric[key]
            else:
                return metric

    def add_metric(self, epoch, key, value):
        self.epochs[epoch].metric[key] = value

    def add_epoch(self, epoch, pred, label, loss=None):
        if loss is not None:
            loss = detach(loss)
        self.epochs[epoch] = Epoch(pred=detach(pred), label=detach(label), loss=loss)

    def insert_buffer(self, epoch, buffer: EpochBuffer):
        self.epochs[epoch] = Epoch.from_buffer(buffer)

    @torch.no_grad()
    def mean_loss(self, epoch):
        return torch.mean(self.epochs[epoch].loss)

    def reset_dataloader(self, *args, **kwargs):
        """
        重新设置dataloader的参数
        Args:
            *args: DataLoader的参数
            **kwargs: DataLoader的参数

        Returns:

        """
        if not hasattr(self.loader, 'dataset'):
            print('data loader do not have dataset, can not be reset', file=sys.stderr)
        dataset = self.loader.dataset
        if hasattr(dataset, 'collate_fn'):
            self.loader = DataLoader(dataset, *args, **kwargs, collate_fn=dataset.collate_fn)
        else:
            self.loader = DataLoader(dataset, *args, **kwargs)

    def get_data_parser(self) -> Union[DataParser, None]:
        """
        获得dataloader的data parser
        Returns:

        """
        if not hasattr(self.loader, 'dataset'):
            print('data loader do not have dataset, can not get data parser', file=sys.stderr)
            return
        if not isinstance(self.loader.dataset, CSVDataset):
            print('dataset is not a CSVDataset, can not get data parser', file=sys.stderr)
            return
        return self.loader.dataset.data_parser
