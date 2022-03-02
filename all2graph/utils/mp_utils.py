from .tqdm_utils import tqdm
try:
    from dgl.multiprocessing import Pool
except ImportError:
    try:
        from torch.multiprocessing import Pool
    except ImportError:
        from multiprocessing import Pool


def mp_run(fn, data, processes=0, chunksize=1, disable=False, postfix=None, **kwargs):
    if processes == 0:
        for item in tqdm(map(fn, data), disable=disable, postfix=postfix, **kwargs):
            yield item
    else:
        with Pool(processes=processes) as pool:
            for item in tqdm(pool.imap(fn, data, chunksize=chunksize), disable=disable, postfix=postfix, **kwargs):
                yield item
