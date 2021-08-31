from typing import Iterable


class Resolver:
    def __init__(self, root_name, **kwargs):
        self.root_name = root_name

    def resolve(self, data: Iterable, progress_bar: bool = False):
        raise NotImplementedError

    __call__ = resolve
