from functools import partial
from .tqdm_utils import tqdm
try:
    from dgl.multiprocessing import Pool
except ImportError:
    try:
        from torch.multiprocessing import Pool
    except ImportError:
        from multiprocessing import Pool


def mp_run(fn, data, fn_kwargs=None, processes=0, chunksize=1, disable=False, postfix=None, **kwargs):
    if fn_kwargs is not None:
        fn = partial(fn, **fn_kwargs)
    with tqdm(data, disable=disable, postfix=postfix, **kwargs) as bar:
        if processes == 0:
            for item in map(fn, data):
                bar.update()
                yield item
        else:
            with Pool(processes=processes) as pool:
                for item in pool.imap(fn, data, chunksize=chunksize):
                    bar.update()
                    yield item
