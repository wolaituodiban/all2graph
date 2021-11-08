import sys
import traceback
from .operator import Operator
from ..utils import rm_ascii


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


class Rename(Operator):
    def __init__(self, old, new, error=True, warning=True):
        super().__init__()
        self.old = old
        self.new = new
        self.error = error
        self.warning = warning

    def __call__(self, obj, **kwargs):
        if self.new in obj:
            if self.error:
                raise KeyError('{} already exists'.format(self.new))
            elif self.warning:
                print('{} already exists'.format(self.new), file=sys.stderr)
        try:
            obj[self.new] = obj[self.old]
            del obj[self.old]
        except (KeyError, ValueError, IndexError, TypeError) as e:
            if self.error:
                raise e
            elif self.warning:
                traceback.print_exc(file=sys.stderr)
        return obj

    def __repr__(self):
        return "{}(old={}, new={})".format(self.__class__.__name__, self.old, self.new)


class RemoveASCII(Operator):
    def __call__(self, obj, **kwargs):
        return rm_ascii(obj)
