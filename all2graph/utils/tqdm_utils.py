import sys


def tqdm(iterable, file=sys.stdout, ascii=True, ncols=100, **kwargs):
    from tqdm import tqdm
    return tqdm(iterable, file=file, ascii=ascii, ncols=ncols, **kwargs)
