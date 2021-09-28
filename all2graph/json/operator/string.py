from .operator import Operator


class Lower(Operator):
    def __init__(self):
        super().__init__()

    def __call__(self, obj, **kwargs):
        if isinstance(obj, str):
            return obj.lower()
        else:
            return obj


class Split(Operator):
    def __init__(self, sep, maxsplit=-1):
        super().__init__()
        self.sep = sep
        self.maxsplit = maxsplit

    def __call__(self, obj: str, **kwargs):
        if isinstance(obj, str):
            return obj.split(sep=self.sep, maxsplit=self.maxsplit)
        else:
            return obj

    def __repr__(self):
        return "{}(sep='{}', maxsplit={})".format(self.__class__.__name__, self.sep, self.maxsplit)


class Cut(Operator):
    def __init__(self):
        super().__init__()

    def __call__(self, obj: str, tokenizer=None, **kwargs):
        if isinstance(obj, str):
            return tokenizer.lcut(obj)
        else:
            return obj
