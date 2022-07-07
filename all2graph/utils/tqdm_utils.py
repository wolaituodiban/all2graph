import sys


def tqdm(iterable, file=sys.stdout, ascii=True, **kwargs):
    from tqdm import tqdm
    return tqdm(iterable, file=file, ascii=ascii, **kwargs)
