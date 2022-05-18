import os
import sys
import traceback
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.errors import ParserError
from .tqdm_utils import tqdm


# def iter_files(inputs, error=True, warning=True):
#     """
#
#     :param inputs: 文件路径、文件夹路径或者Iterable的任意嵌套
#     :param error: 如果inputs包含的内容不是文件路径、文件夹路径或者Iterable，那么会报错
#     :param warning: 如果inputs包含的内容不是文件路径、文件夹路径或者Iterable，那么会报warning
#     :return: 遍历所有文件路径的生成器
#     """
#     if isinstance(inputs, str):
#         if os.path.exists(inputs):
#             if os.path.isdir(inputs):
#                 for dirpath, dirnames, filenames in os.walk(inputs):
#                     for filename in filenames:
#                         yield os.path.join(dirpath, filename)
#             else:
#                 yield inputs
#         elif error:
#             raise ValueError('path {} dose not exists'.format(inputs))
#         elif warning:
#             print('path {} dose not exists'.format(inputs), file=sys.stderr)
#     elif isinstance(inputs, Iterable):
#         for item in inputs:
#             for path in iter_files(item, error=error, warning=warning):
#                 yield path
#     elif error:
#         raise ValueError('path {} dose not exists'.format(inputs))
#     elif warning:
#         print('path {} dose not exists'.format(inputs), file=sys.stderr)


def iter_files(inputs, error=True, warning=True):
    """

    :param inputs: 文件路径、文件夹路径或者Iterable的任意嵌套
    :param error: 如果inputs包含的内容不是文件路径、文件夹路径或者Iterable，那么会报错
    :param warning: 如果inputs包含的内容不是文件路径、文件夹路径或者Iterable，那么会报warning
    :return: 遍历所有文件路径的生成器
    """
    if isinstance(inputs, str):
        if os.path.exists(inputs):
            if os.path.isdir(inputs):
                for path in iter_files(
                        [os.path.join(inputs, path) for path in os.listdir(inputs)], error=error, warning=warning
                ):
                    yield path
            else:
                yield inputs
        elif error:
            raise ValueError('path {} dose not exists'.format(inputs))
        elif warning:
            print('path {} dose not exists'.format(inputs), file=sys.stderr)
    elif isinstance(inputs, Iterable):
        for item in inputs:
            for path in iter_files(item, error=error, warning=warning):
                yield path
    elif error:
        raise ValueError('path {} dose not exists'.format(inputs))
    elif warning:
        print('path {} dose not exists'.format(inputs), file=sys.stderr)


def iter_csv(inputs, chunksize, error=True, warning=True, concat_chip=True, recurse=True, **kwargs):
    """

    Args:
        inputs: panda DataFrame或者"文件路径、文件夹路径或者Iterable的任意嵌套"
        chunksize:
        error: 发生错误时会raise ValueError
        warning: 发生错误时会打印错误信息
        concat_chip: 拼接小于chunksize的chunk，保证（除最后一个）所有chunk的大小都是chunksize
        recurse: 是否递归inputs，寻找包含的所有路径；如果False，那么默认inputs是一个list of path
        **kwargs: 传递给read_csv的参数

    Returns:
        dataframe分片生成器
    """
    if isinstance(inputs, pd.DataFrame):
        for i in range(int(np.ceil(inputs.shape[0] / chunksize))):
            yield inputs.iloc[chunksize * i:chunksize * (i + 1)]
    else:
        buffer = pd.DataFrame()
        if recurse:
            inputs = iter_files(inputs, error=error, warning=warning)
        for path in inputs:
            try:
                for chunk in pd.read_csv(path, chunksize=chunksize, **kwargs):
                    if concat_chip:
                        buffer = pd.concat([buffer, chunk])
                    else:
                        yield chunk
                    if buffer.shape[0] >= chunksize:
                        yield buffer.iloc[:chunksize]
                        buffer = buffer.iloc[chunksize:]
            except (ParserError, ValueError):
                if error:
                    raise ValueError('read "{}" encountered error'.format(path))
                elif warning:
                    print('read "{}" encountered error'.format(path), file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
        if buffer.shape[0] > 0:
            yield buffer


def split_csv(
        src, dst, chunksize, disable=False, zip=True, error=True, warning=True, concat_chip=True, sel_cols=None,
        drop_cols=None, **kwargs):
    """

    Args:
        src: panda DataFrame或者"文件路径、文件夹路径或者Iterable的任意嵌套"
        dst: 保存分片csv文件的目录
        chunksize:
        disable:
        zip: 是否压缩
        error: 发生错误时会raise ValueError
        warning: 发生错误时会打印错误信息
        concat_chip: 拼接小于chunksize的chunk，保证（除最后一个）所有chunk的大小都是chunksize
        sel_cols: 需要保存到元信息dataframe的列名
        drop_cols: 需要去掉的列，只在meta_col为None时生效
        **kwargs:

    Returns:
        包含文件路径等元信息的dataframe
    """
    assert sel_cols is None or isinstance(sel_cols, list)
    meta_dfs = []
    if os.path.exists(dst):
        raise ValueError('{} already exists'.format(dst))
    os.mkdir(dst)
    chunk_iter = enumerate(
        iter_csv(src, chunksize=chunksize, error=error, warning=warning, concat_chip=concat_chip, **kwargs))
    for i, chunk in tqdm(chunk_iter, disable=disable, postfix='spliting csv'):
        if zip:
            to_file = os.path.join(dst, '{}.{}'.format(i, 'zip'))
        else:
            to_file = os.path.join(dst, '{}.{}'.format(i, 'csv'))
        chunk.to_csv(to_file)
        if sel_cols is not None:
            path_df = chunk[sel_cols]
            path_df['path'] = to_file
        elif drop_cols is not None:
            path_df = chunk.drop(columns=drop_cols)
            path_df['path'] = to_file
        else:
            path_df = pd.DataFrame({'path': [to_file] * chunk.shape[0]})
        meta_dfs.append(path_df)
    path_df = pd.concat(meta_dfs)
    path_df.to_csv(dst+'_path.{}'.format('zip' if zip else 'csv'), index=False)
    return path_df


def timestamp_convertor(x):
    return x.replace('T', ' ')[:23]
