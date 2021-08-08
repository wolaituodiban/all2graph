from typing import Iterable, Union
import pandas as pd
from .meta_graph import MetaGraph


class JsonGraph(MetaGraph):
    """解析json，并生成表述json结构的元图"""
    @classmethod
    def from_data(cls, data: Union[pd.DataFrame, Iterable] = None, jsons: Union[str, Iterable] = None,
                  times: Union[str, Iterable] = None, ids: Union[str, Iterable] = None, **kwargs):
        """
        由json序列生成原图
        :param data:
        :param jsons:
        :param times:
        :param ids:
        :param kwargs:
        :return:
        """
        jsons = jsons or data[jsons]
        times = times or data[times]
        if ids is None:
            ids = list(range(len(jsons)))
        else:
            ids = data[ids]
        assert len(jsons) == len(times) == len(ids)

        # todo