from typing import Dict, Tuple, List, Set

import numpy as np
import pandas as pd


from ..graph import RawGraph
from ..preserves import NULL, PRESERVED_WORDS, READOUT, KEY, TARGET, META
from ..meta import MetaInfo, MetaNumber
from ..utils import Tokenizer, default_tokenizer
from ..meta_struct import MetaStruct


class RawGraphParser(MetaStruct):
    def __init__(self, meta_numbers: Dict[str, MetaNumber], strings: list, keys: List[str],
                 edge_type: Set[Tuple[str, str]], targets: List[str] = None, tokenizer: Tokenizer = default_tokenizer):
        """
        Graph与dgl.DiGraph的转换器
        :param meta_numbers: 数值分布
        :param strings: 字符串编码字典
        :param keys: 如果是dict，那么dict的元素必须是list，代表name的分词
        :param edge_type:
        :param targets:
        :parma meta_mode: 如果是True，那么graph_to_dgl会生成一个元图和一个值图，否则只生成一个值图
        """
        super().__init__(initialized=True)
        self.meta_numbers = {k: MetaNumber.from_json(v.to_json()) for k, v in meta_numbers.items()}
        self.tokenizer = tokenizer
        self.targets = sorted(list(targets or []))
        self.key_mapper = {k: i for i, k in enumerate(sorted(set(keys + self.targets)))}

        self.etype_mapper = {t: i for i, t in enumerate(sorted(set(edge_type)))}
        for target in self.targets:
            if (READOUT, target) not in self.etype_mapper:
                self.etype_mapper[(READOUT, target)] = len(self.etype_mapper)

        all_words = PRESERVED_WORDS + sorted(strings)
        for key in keys + self.targets:
            all_words += self.tokenizer.lcut(key)
        self.string_mapper = {}
        for word in (word.lower() for word in all_words):
            if word not in self.string_mapper:
                self.string_mapper[word] = len(self.string_mapper)

        assert all(i == self.string_mapper[w] for i, w in enumerate(PRESERVED_WORDS))
        assert set(map(type, self.string_mapper)) == {str}
        assert len(self.string_mapper) == len(set(self.string_mapper.values())), (
            len(self.string_mapper), len(set(self.string_mapper.values()))
        )
        assert len(self.string_mapper) == max(list(self.string_mapper.values())) + 1

    @property
    def num_strings(self):
        return len(self.string_mapper)

    @property
    def num_targets(self):
        return len(self.targets)

    @property
    def num_keys(self):
        return len(self.key_mapper)

    @property
    def num_numbers(self):
        return len(self.meta_numbers)

    @property
    def num_etypes(self):
        return len(self.etype_mapper)

    @property
    def reverse_string_mapper(self):
        return {v: k for k, v in self.string_mapper.items()}

    @property
    def target_symbol(self):
        return self.encode_string([TARGET])

    def get_quantiles(self, name, p, **kwargs):
        if name in self.meta_numbers:
            return self.meta_numbers[name].value_ecdf.get_quantiles(p, **kwargs)
        else:
            return np.full_like(p, np.nan)

    def get_probs(self, name, q, **kwargs):
        if name in self.meta_numbers:
            return self.meta_numbers[name].value_ecdf.get_probs(q, **kwargs)
        else:
            return np.full_like(q, np.nan)

    def encode_string(self, inputs: list) -> List[int]:
        output = [self.string_mapper.get(str(x).lower(), self.string_mapper[NULL]) for x in inputs]
        return output

    def encode_key(self, inputs: list) -> List[int]:
        # 为了让模型的错误调用会报错，此处故意将不存在的key填了一个会越界的值
        output = [self.key_mapper.get(key, -self.num_keys-1) for key in inputs]
        return output

    def encode_edge_key(self, inputs: list) -> List[int]:
        # 为了让模型的错误调用会报错，此处故意将不存在的key填了一个会越界的值
        output = [self.etype_mapper.get(etype, -self.num_etypes-1) for etype in inputs]
        return output

    def normalize(self, key, value) -> np.ndarray:
        numbers = pd.to_numeric(pd.Series(value).values, errors='coerce')
        unique_names, inverse_indices = np.unique(key, return_inverse=True)
        masks = np.eye(unique_names.shape[0], dtype=bool)[inverse_indices]
        for i in range(unique_names.shape[0]):
            numbers[masks[:, i]] = self.get_probs(unique_names[i], numbers[masks[:, i]])
        return numbers

    def parse(self, graph: RawGraph):
        """

        :param graph:
        :return:
        """
        from all2graph.graph.graph import Graph

        graph = graph.add_targets(self.targets)
        meta_graph, meta_node_id, meta_edge_id = graph.meta_graph(self.tokenizer)
        return Graph(
            meta_graph=meta_graph, graph=graph, meta_key=self.encode_key(meta_graph.key),
            meta_value=self.encode_string(meta_graph.value), meta_symbol=self.encode_string(meta_graph.symbol),
            meta_component_id=meta_graph.component_id, meta_edge_key=self.encode_edge_key(meta_graph.edge_key),
            value=self.encode_string(graph.value), number=self.normalize(graph.key, graph.value),
            symbol=self.encode_string(graph.symbol), meta_node_id=meta_node_id, meta_edge_id=meta_edge_id
        )

    def gen_param_graph(self, param_names):
        from all2graph.graph.param import ParamGraph
        param_mapper = {}
        raw_graph = RawGraph()
        for name in param_names:
            if name not in param_mapper:
                param_id = raw_graph.insert_node(0, key=KEY, value=KEY, self_loop=True, symbol=KEY)
                param_mapper[name] = param_id
                pre_ids = [param_id]
                for token in self.tokenizer.cut(name):
                    if token not in raw_graph.value:
                        token_id = raw_graph.insert_node(0, key=META, value=token, self_loop=False, symbol=META)
                    else:
                        token_id = raw_graph.value.index(token)
                    raw_graph.insert_edges([token_id] * len(pre_ids), pre_ids, bidirection=True)
                    pre_ids.append(token_id)
            else:
                param_id = param_mapper[name]
                raw_graph.key[param_id] = KEY
                raw_graph.symbol[param_id] = KEY
        raw_graph.drop_duplicated_edges()
        return ParamGraph(graph=raw_graph, value=self.encode_string(raw_graph.value), mapper=param_mapper)

    def __eq__(self, other, debug=False):
        if not super().__eq__(other):
            if debug:
                print('super not equal')
            return False
        if self.targets != other.targets:
            if debug:
                print('targets not equal')
            return False
        if self.meta_numbers != other.meta_numbers:
            if debug:
                print('meta_numbers not equal')
            return False
        if self.string_mapper != other.string_mapper:
            if debug:
                print('string_mapper not equal')
            return False
        if self.key_mapper != other.key_mapper:
            if debug:
                print('key_mapper not equal')
            return False
        if self.etype_mapper != other.etype_mapper:
            if debug:
                print('etype_mapper not equal')
            return False
        return True

    def to_json(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_json(cls, obj: dict):
        raise NotImplementedError

    @classmethod
    def reduce(cls, parsers: list, tokenizer=None, weights=None, num_bins=None):
        meta_numbers = {}
        strings = []
        keys = []
        edge_type = set()
        targets = []
        tokenizer = tokenizer or parsers[0].tokenizer
        for parser in parsers:
            for k, v in parser.meta_numbers.items():
                if k in meta_numbers:
                    meta_numbers[k] = MetaNumber.reduce([meta_numbers[k], v], weights=weights, num_bins=num_bins)
                else:
                    meta_numbers[k] = v
            strings += list(parser.string_mapper)
            keys += list(parser.key_mapper)
            edge_type = edge_type.union(parser.etype_mapper)
            targets += list(parser.targets)
        return cls(
            meta_numbers=meta_numbers, strings=strings, keys=keys, edge_type=edge_type, targets=targets,
            tokenizer=tokenizer)

    def extra_repr(self) -> str:
        return 'num_numbers={}, num_strings={}, num_keys={}, num_etype={}, targets={}'.format(
            self.num_numbers, self.num_strings, self.num_keys, self.num_etypes, self.targets
        )

    @classmethod
    def from_data(cls, meta_info: MetaInfo, min_df=0, max_df=1, top_k=None, top_method='mean_tfidf',
                  targets=None, tokenizer: Tokenizer = default_tokenizer):
        """

        :param meta_info:
        :param min_df: 字符串最小文档平吕
        :param max_df: 字符串最大文档频率
        :param top_k: 选择前k个字符串
        :param top_method: 'max_tfidf', 'mean_tfidf', 'max_tf', 'mean_tf', 'max_tc', mean_tc'
        :param targets:
        :param tokenizer:
        """
        strings = [k for k, df in meta_info.meta_string.doc_freq().items() if min_df <= df <= max_df]
        if top_k is not None:
            if top_method == 'max_tfidf':
                strings = [(k, v.max) for k, v in meta_info.meta_string.tf_idf_ecdf(strings).items()]
            elif top_method == 'mean_tfidf':
                strings = [(k, v.mean) for k, v in meta_info.meta_string.tf_idf_ecdf(strings).items()]
            elif top_method == 'max_tf':
                strings = [(k, v.max) for k, v in meta_info.meta_string.term_freq_ecdf.items() if k in strings]
            elif top_method == 'mean_tf':
                strings = [(k, v.mean) for k, v in meta_info.meta_string.term_freq_ecdf.items() if k in strings]
            elif top_method == 'max_tc':
                strings = [(k, v.max) for k, v in meta_info.meta_string.term_count_ecdf.items() if k in strings]
            elif top_method == 'mean_tc':
                strings = [(k, v.mean) for k, v in meta_info.meta_string.term_count_ecdf.items() if k in strings]
            else:
                raise ValueError(
                    "top_method只能是('max_tfidf', 'mean_tfidf', 'max_tf', 'mean_tf', 'max_tc', mean_tc')其中之一"
                )
            strings = sorted(strings, key=lambda x: x[1])
            strings = [k[0] for k in strings[:top_k]]

        meta_numbers = {key: ecdf for key, ecdf in meta_info.meta_numbers.items() if ecdf.value_ecdf.mean_var[1] > 0}

        all_keys = list(meta_info.meta_name)
        return cls(keys=all_keys, meta_numbers=meta_numbers, strings=strings, tokenizer=tokenizer, targets=targets,
                   edge_type=meta_info.edge_type)
