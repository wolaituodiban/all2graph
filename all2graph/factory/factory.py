from typing import Union

from ..graph import RawGraph
from ..info import MetaInfo
from ..meta_struct import MetaStruct
from ..parsers import DataParser, GraphParser, ParserWrapper


def _verify_kwargs(func, *args, kwargs):
    unexpected = []
    for k, v in kwargs.items():
        try:
            func(*args, **{k: v})
        except TypeError:
            unexpected.append(k)
        except:
            pass
    return unexpected


class Factory(MetaStruct):
    """Factory"""
    def __init__(
            self, data_parser: DataParser, meta_info_config: dict = None, graph_parser_config: dict = None):
        """

        Args:
            data_parser:
            meta_info_config:
                num_bins: 统计各种分布时的分箱数量
            graph_parser_config:
                min_df: 字符串最小文档频率
                max_df: 字符串最大文档频率
                top_k: 选择前k个字符串
                top_method: 'max_tfidf', 'mean_tfidf', 'max_tf', 'mean_tf', 'max_tc', mean_tc'
                scale_method: 'prob' or 'minmax_scale'
                scale_kwargs: 具体见MetaNumber.get_probs和MetaNumber.minmax_scale
        """

        super().__init__(initialized=True)
        if meta_info_config is not None:
            unexp_args = _verify_kwargs(MetaInfo.from_data, RawGraph(), kwargs=meta_info_config)
            assert len(unexp_args) == 0, 'meta_info_config got unexpected keyword argument {}'.format(unexp_args)
        if graph_parser_config is not None:
            unexp_args = _verify_kwargs(GraphParser.from_data, None, kwargs=graph_parser_config)
            assert len(unexp_args) == 0, 'raw_graph_parser_config got unexpected keyword argument {}'.format(unexp_args)

        self.data_parser = data_parser
        self.meta_info_config = meta_info_config or {}
        self.meta_info: Union[MetaInfo, None] = None
        self.graph_parser_config = graph_parser_config or {}
        self.graph_parser: Union[GraphParser, None] = None

    @property
    def parser_wrapper(self) -> ParserWrapper:
        return ParserWrapper(self.data_parser, self.graph_parser)

    def analyse(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def produce_dataloader(self):
        raise NotImplementedError

    def produce_model(self):
        raise NotImplementedError

    def extra_repr(self) -> str:
        return 'data_parser: {}\ngraph_parser: {}'.format(
            self.data_parser, self.graph_parser
        )

    def __eq__(self, other):
        return super().__eq__(other) \
               and self.graph_parser == other.graph_parser \
               and self.data_parser == other.graph_parser \
               and self.graph_parser_config == other.graph_parser_config \
               and self.meta_info_config == other.meta_info_config

    @classmethod
    def from_data(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_json(cls, obj: dict):
        raise NotImplementedError

    @classmethod
    def batch(cls, structs, weights=None, **kwargs):
        raise NotImplementedError

    def to_json(self) -> dict:
        raise NotImplementedError

