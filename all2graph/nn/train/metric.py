from inspect import isfunction

from .callback import CallBack
from .history import History
from ..utils import to_numpy


class Metric(CallBack):
    def __init__(self, func, name):
        """

        Args:
            func: label_first
            name:
        """
        self.func = func
        self.name = name

    def __repr__(self):
        if isfunction(self.func):
            func_repr = self.func.__name__
        else:
            func_repr = self.func
        return '{}(func={}, name="{}")'.format(self.__class__.__name__, func_repr, self.name)

    def __call__(self, trainer, history: History, epoch: int):
        label = history.get_label(epoch)
        pred = history.get_pred(epoch)
        if label is not None and pred is not None:
            metric = self.func(to_numpy(label), to_numpy(pred))
            history.add_metric(epoch, key=self.name, value=metric)
