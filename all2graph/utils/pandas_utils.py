import numpy as np
import pandas as pd


def dataframe_chunk_iter(data, chunksize=64, **kwargs):
    if isinstance(data, pd.DataFrame):
        for i in range(int(np.ceil(data.shape[0] / chunksize))):
            yield data.iloc[chunksize*i:chunksize*(i+1)]
    else:
        for chunk in pd.read_csv(data, chunksize=chunksize, **kwargs):
            yield chunk
