from copy import deepcopy
from typing import Callable

from .callback import CallBack
from .history import History
from ...globals import EPSILON


class EarlyStop(CallBack):
    def __init__(self, rounds, higher: bool, tol=EPSILON, loader_id=0, json_path: Callable = None):
        """

        Args:
            rounds (int): 停止的轮数
            higher (bool): metric是否越大越好
            tol (float): metric没有变好的容忍度
            loader_id: 以哪个数据集为准，None表示train loader
            json_path: 如果metric是一个复杂结构，那么这个参数将定义处理json的方法，默认metric是个float
        """
        self.rounds = rounds
        self.higher = higher
        self.tol = tol
        self.loader_id = loader_id
        self.json_path_tree = json_path

        self._best_metric = None
        self._best_epoch = None
        self._bset_state = None

    @property
    def sign(self):
        return 2 * self.higher - 1

    def __repr__(self):
        if self.json_path_tree:
            json_path_tree_repr = '\n'.join('    '+x for x in str(self.json_path_tree).split('\n'))
        else:
            json_path_tree_repr = None
        return '{}(\n  rounds={},\n  higher={},\n  tol={},\n  loader_id={},\n  json_path_tree=(\n{}\n  )\n),\n  best_epoch={}'.format(
            self.__class__.__name__, self.rounds, self.higher, self.tol, self.loader_id, json_path_tree_repr, self._best_epoch
        )

    def __call__(self, trainer, _, epoch: int) -> bool:
        """

        Args:
            trainer:
            _:
            epoch:

        Returns:
            True是停止信号
        """
        if self.loader_id is None:
            history: History = trainer.train_history
        else:
            history: History = trainer.valid_history[self.loader_id]

        metric = history.last.metric
        if self.json_path_tree is not None:
            metric = self.json_path_tree(deepcopy(metric))

        if self._best_metric is None or \
                (metric != self._best_metric and (metric - self._best_metric) * self.sign > self.tol):
            self._best_metric = metric
            self._best_epoch = epoch
            self._best_state_dict = trainer.module.state_dict()

        signal = (epoch - self._best_epoch) > self.rounds
        if signal:
            trainer.module.load_state_dict(self._best_state_dict)
            trainer._current_epoch = self._best_epoch
        return signal
