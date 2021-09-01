from typing import Iterable


class DataParser:
    def __init__(self, root_name, **kwargs):
        self.root_name = root_name

    def parse(self, data: Iterable, progress_bar: bool = False):
        raise NotImplementedError

    __call__ = parse
