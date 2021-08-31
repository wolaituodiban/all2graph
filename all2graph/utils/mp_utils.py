class MpMapFuncWrapper:
    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs

    def __call__(self, x):
        return self.func(**x, **self.kwargs)
