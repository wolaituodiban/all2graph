import sys
import traceback
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from pandas.errors import ParserError
from torch.utils.data import Dataset as TorchDataset

from ..graph import RawGraph
from ..parsers import DataParser
from ..utils import progress_wrapper, iter_files


class Dataset(TorchDataset):
    def __init__(
            self, src, parser: DataParser, target_cols, chunksize=64, shuffle=False, disable=True, error=True,
            warning=True, **kwargs):
        self.parser = parser
        self.target_cols = target_cols or []
        self.kwargs = kwargs

        paths: List[Tuple[str, int]] = []
        for path in progress_wrapper(
                iter_files(src, error=error, warning=warning), disable=disable, postfix='checking files'):
            try:
                df = self.read_csv(path)
                row_nums = list(range(df.shape[0]))
                if shuffle:
                    np.random.shuffle(row_nums)
                for row_num in row_nums:
                    paths.append((path, row_num))
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except ParserError:
                if error:
                    raise ValueError('read "{}" encountered error'.format(path))
                elif warning:
                    print('read "{}" encountered error'.format(path), file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)

        self.paths: List[Dict[str, List[int]]] = []
        for i in range(0, len(paths), chunksize):
            temp = {}
            for path, row_num in paths[i:i + chunksize]:
                if path in temp:
                    temp[path].append(row_num)
                else:
                    temp[path] = [row_num]
            self.paths.append(temp)
        # 转化成pandas类型，似乎可以减小datalodaer多进程的开销
        self.paths = pd.Series(self.paths)

    def read_csv(self, path):
        return pd.read_csv(path, **self.kwargs)

    @property
    def chunksize(self):
        return sum(map(len, self.paths[0].values()))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item) -> Tuple[RawGraph, Dict[str, torch.Tensor]]:
        dfs = []
        for path, row_nums in self.paths.iloc[item].items():
            df = self.read_csv(path)
            df = df.iloc[row_nums]
            dfs.append(df)
        df = pd.concat(dfs)
        graph = self.parser.parse(df, progress_bar=False)[0]
        label = self.parser.gen_targets(df, self.target_cols)
        return graph, label
