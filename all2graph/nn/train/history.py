import sys
from typing import Dict

import torch
from torch.utils.data import Dataset as _Dataset, DataLoader
from ..utils import detach, default_collate
from ...data import ParserDataset
from ...parsers import ParserWrapper


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

    def delete_pred_and_label(self):
        self.pred = None
        self.label = None


class History:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.epochs: Dict[int, Epoch] = {}

    @property
    def dataset(self) -> _Dataset:
        if hasattr(self.loader, 'dataset'):
            return self.loader.dataset

    @property
    def parser(self) -> ParserWrapper:
        dataset = self.dataset
        if isinstance(dataset, ParserDataset):
            return dataset.parser

    @property
    def num_epochs(self):
        return len(self.epochs)

    def pop(self):
        min_epoch = min(list(self.epochs))
        output = self.epochs[min_epoch]
        del self.epochs[min_epoch]
        return output

    def delete_history(self, epochs):
        """

        Args:
            epochs: 保留多少个epoch

        Returns:

        """
        max_epoch = max(list(self.epochs))
        for i, epoch in self.epochs.items():
            if i + epochs <= max_epoch:
                epoch.delete_pred_and_label()

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

    def get_loss(self, epoch) -> torch.Tensor:
        if epoch in self.epochs:
            return self.epochs[epoch].loss

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
