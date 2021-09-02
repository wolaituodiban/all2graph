import os
import numpy as np
import pandas as pd
from .toad_utils import progress_wrapper


def dataframe_chunk_iter(data, chunksize, **kwargs):
    if isinstance(data, pd.DataFrame):
        for i in range(int(np.ceil(data.shape[0] / chunksize))):
            yield data.iloc[chunksize*i:chunksize*(i+1)]
    else:
        for chunk in pd.read_csv(data, chunksize=chunksize, **kwargs):
            yield chunk


def split_csv(path, dir_path, chunksize, disable=True, **kwargs):
    chunk_iter = enumerate(dataframe_chunk_iter(path, chunksize=chunksize, **kwargs))
    file_name = os.path.split(path)[-1]
    for i, chunk in progress_wrapper(chunk_iter, disable=disable, postfix='spliting csv'):
        chunk.to_csv(os.path.join(dir_path, '{}.{}'.format(file_name, i)))
