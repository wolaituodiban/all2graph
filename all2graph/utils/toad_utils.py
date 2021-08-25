import sys
import toad
from toad.utils.progress import Progress


def progress_wrapper(iterable, size=None, disable=False, suffix=None):
    if disable or isinstance(iterable, Progress):
        return iterable
    else:
        output = Progress(iterable, size=size)
        if suffix is not None:
            output.suffix = suffix
        if toad.version.__version__ <= '0.0.65' and output.size is None:
            output.size = sys.maxsize
        return output
