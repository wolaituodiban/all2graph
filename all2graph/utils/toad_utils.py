import sys

try:
    from tqdm import tqdm

    def progress_wrapper(iterable, total=None, disable=False, postfix=None, file=sys.stdout, ascii=True, **kwargs):
        return tqdm(iterable, total=total, disable=disable, postfix=postfix, file=file, ascii=ascii, **kwargs)


except ImportError:
    try:
        import toad
        from toad.utils.progress import Progress

        def progress_wrapper(iterable, total=None, disable=False, postfix=None, **kwargs):
            if disable or isinstance(iterable, Progress):
                return iterable
            else:
                output = Progress(iterable, size=total)
                if postfix is not None:
                    output.suffix = postfix
                if toad.version.__version__ <= '0.0.65' and output.size is None:
                    output.size = sys.maxsize
                return output

    except ImportError:
        def progress_wrapper(iterable, **kwargs):
            return iterable
