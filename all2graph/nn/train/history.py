from typing import Dict

from torch.utils.data import DataLoader
from ..utils import detach, default_collate


class Epoch:
    def __init__(self, pred, label, loss=None):
        self.pred = pred
        self.label = label
        self.loss = loss
        self.metric = {}


class History:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.epochs: Dict[int, Epoch] = {}
        self._current_pred = []
        self._current_label = []
        self._current_loss = []

    @property
    def num_epochs(self):
        return len(self.epochs)

    def log(self, pred, label, loss):
        self._current_pred.append(detach(pred))
        self._current_label.append(detach(label))
        self._current_loss.append(detach(loss))

    def collate(self, epoch):
        self.epochs[epoch] = Epoch(
            pred=default_collate(self._current_pred),
            label=default_collate(self._current_label),
            loss=default_collate(self._current_loss)
        )
        self._current_pred = []
        self._current_loss = []
        self._current_label = []

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
        return self.epochs[epoch].label

    def get_pred(self, epoch):
        return self.epochs[epoch].pred

    def get_metric(self, epoch, key=None):
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
