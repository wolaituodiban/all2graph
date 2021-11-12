import sys


def tqdm(iterable, file=sys.stdout, ascii=True, **kwargs):
    try:
        from tqdm import tqdm
        return tqdm(iterable, file=file, ascii=ascii, **kwargs)
    except ImportError:
        return iterable
