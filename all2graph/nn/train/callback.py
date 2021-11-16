from .history import History


class CallBack:
    def __call__(self, trainer, history: History, epoch: int):
        raise NotImplementedError
