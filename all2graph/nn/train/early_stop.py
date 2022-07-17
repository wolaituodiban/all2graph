"""提前停止模块"""
from copy import deepcopy
from typing import Callable

from .callback import CallBack
from .history import History
from ...globals import EPSILON


class EarlyStop(CallBack):
    """提前停止类"""
    def __init__(self, rounds, higher: bool, tol=EPSILON, loader_id=0, fn: Callable = None):
        """

        Args:
            rounds (int): 停止的轮数
            higher (bool): metric是否越大越好
            tol (float): metric没有变好的容忍度
            loader_id: 以哪个数据集为准, None表示train loader
            fn: 如果metric是一个复杂结构, 那么这个参数将定义处理json的方法, 默认metric是个float
        """
        self.rounds = rounds
        self.higher = higher
        self.tol = tol
        self.loader_id = loader_id
        self.fn = fn

        self._best_metric = None
        self._best_epoch = None
        # self._best_state_dict = None

    @property
    def sign(self):
        """当metric越高越好时, 输出1; 否则, 输出0"""
        return 2 * self.higher - 1

    @property
    def best_metric(self):
        """最佳metric"""
        return deepcopy(self._best_metric)

    @property
    def best_epoch(self):
        """最佳epoch"""
        return deepcopy(self.best_epoch)

    def __repr__(self):
        if self.fn:
            fn_expr = '\n'.join('    '+x for x in str(self.fn).split('\n'))
        else:
            fn_expr = None

        output = '{}(\n  rounds={},\n  higher={},\n  tol={},\n  loader_id={},\n  best_epoch={},\n'
        output = output.format(
            self.__class__.__name__, self.rounds, self.higher, self.tol, self.loader_id,
             self._best_epoch)
        output += f'  fn=(\n{fn_expr}\n  )\n)'
        return output

    def __call__(self, trainer, _, epoch: int) -> bool:
        """

        Args:
            trainer:
            _:
            epoch:

        Returns:
            True是停止信号
        """
        history: History = trainer.get_history(self.loader_id)

        metric = deepcopy(history.last.metric)
        if self.fn is not None:
            metric = self.fn(metric)

        if self._best_metric is None or \
                (metric != self._best_metric
                 and (metric - self._best_metric) * self.sign > self.tol):
            self._best_metric = metric
            self._best_epoch = epoch
            # self._best_state_dict = deepcopy(trainer.module.state_dict())
        msg = [
            f'current_epoch={epoch}',
            f'current_metric={metric:.3f}',
            f'best_epoch={self._best_epoch}',
            f'best_metric={self._best_metric:.3f}'
        ]
        print(', '.join(msg))

        signal = self.rounds is not None and (epoch - self._best_epoch) >= self.rounds
        # if signal:
        #     trainer.module.load_state_dict(self._best_state_dict)
        #     trainer._current_epoch = self._best_epoch

        return signal
