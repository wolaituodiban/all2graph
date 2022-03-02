from typing import Dict

import numpy as np
import pandas as pd

from .number_info import NumberInfo, _NumberReducer
from .token_info import TokenInfo, _TokenReducer
from .meta_info import MetaInfo
from ..stats import ECDF, _ECDFReducer
from ..utils import tqdm, mp_run


class GraphInfo(MetaInfo):
    def __init__(self, number_infos: Dict[str, NumberInfo], token_infos: Dict[str, TokenInfo],
                 key_counts: Dict[str, ECDF], **kwargs):
        super().__init__(**kwargs)
        self.number_infos = number_infos
        self.token_infos = token_infos
        self.key_counts = key_counts

    def __eq__(self, other, debug=False):
        if not super().__eq__(other):
            if debug:
                print('super not equal')
            return False
        if self.number_infos != other.number_infos:
            if debug:
                print('number_infos not equal')
            return False
        if self.token_infos != other.token_infos:
            if debug:
                print('token_infos not equal')
            return False
        if self.key_counts != other.key_counts:
            if debug:
                print('key_counts not equal')
            return False
        return True

    def dictionary(self, min_df=0, max_df=1, top_k=None, top_method='mean_tfidf',) -> Dict[str, int]:
        dictionary = [k for k, df in self.doc_freq.items() if min_df <= df <= max_df]
        if top_k is not None:
            if top_method == 'max_tfidf':
                dictionary = [(k, v.max) for k, v in self.tf_idf.items() if k in dictionary]
            elif top_method == 'mean_tfidf':
                dictionary = [(k, v.mean) for k, v in self.tf_idf.items() if k in dictionary]
            elif top_method == 'max_tf':
                dictionary = [(k, v.max) for k, v in self.tf_idf.items() if k in dictionary]
            elif top_method == 'mean_tf':
                dictionary = [(k, v.mean) for k, v in self.tf_idf.items() if k in dictionary]
            elif top_method == 'max_tc':
                dictionary = [(k, v.max) for k, v in self.tf_idf.items() if k in dictionary]
            elif top_method == 'mean_tc':
                dictionary = [(k, v.mean) for k, v in self.tf_idf.items() if k in dictionary]
            else:
                raise ValueError(
                    "top_method只能是('max_tfidf', 'mean_tfidf', 'max_tf', 'mean_tf', 'max_tc', mean_tc')其中之一"
                )
            dictionary = sorted(dictionary, key=lambda x: x[1])
            dictionary = [k[0] for k in dictionary[:top_k]]

        for key in self.key_counts:
            if isinstance(key, str):
                dictionary.append(key)
            elif isinstance(key, tuple):
                dictionary += list(key)

        dictionary = set(dictionary)
        return {k: i for i, k in enumerate(dictionary)}

    @property
    def numbers(self) -> Dict[str, ECDF]:
        return {key: info.value for key, info in self.number_infos.items()}

    @property
    def num_keys(self):
        return len(self.key_counts)

    @property
    def num_tokens(self):
        return len(self.token_infos)

    @property
    def num_numbers(self):
        return len(self.number_infos)

    @property
    def doc_freq(self) -> Dict[str, float]:
        return {token: info.doc_freq for token, info in self.token_infos.items()}

    @property
    def tf_idf(self) -> Dict[str, ECDF]:
        return {token: info.tf_idf for token, info in self.token_infos.items()}

    def to_json(self) -> dict:
        output = super().to_json()
        output['number_infos'] = {k: v.to_json() for k, v in self.number_infos.items()}
        output['token_infos'] = {k: v.to_json() for k, v in self.token_infos.items()}
        return output

    @classmethod
    def from_json(cls, obj):
        obj = dict(obj)
        obj['number_infos'] = {k: NumberInfo.from_json(v) for k, v in obj['number_infos'].items()}
        obj['token_infos'] = {k: TokenInfo.from_json(v) for k, v in obj['token_infos'].items()}
        obj['key_count'] = {k: ECDF.from_json(v) for k, v in obj['key_count'].items()}
        return super().from_json(obj)

    @classmethod
    def from_data(cls, sample_ids, keys, values, num_bins=None, disable=True):
        data_df = pd.DataFrame({'sid': sample_ids, 'key': keys, 'token': values})
        # 分离数值型数据和字符串数据
        data_df['number'] = pd.to_numeric(data_df['token'], errors='coerce')
        data_df.loc[pd.notna(data_df['number']), 'token'] = None
        data_df['key_copy'] = data_df['key']
        data_df['number_copy'] = data_df['number']
        data_df['token_copy'] = data_df['token']

        # 统计计数
        count_df = pd.DataFrame({'count': data_df['sid'].value_counts()})
        temp_df = data_df.pivot_table(values='token', index='sid', columns='token_copy', aggfunc='count')
        count_df[[('token', x) for x in temp_df]] = temp_df
        temp_df = data_df.pivot_table(values=['number', 'key'], index='sid', columns='key_copy', aggfunc='count')
        count_df[temp_df.columns] = temp_df
        count_df = count_df.loc[:, count_df.sum() > 0]
        count_df = count_df.fillna(0)
        del temp_df

        token_infos = {}
        number_infos = {}
        key_counts = {}
        for col, series in tqdm(count_df.iteritems(), disable=disable, postfix='constructing {}'.format(cls.__name__)):
            if not isinstance(col, tuple):
                continue
            if col[0] == 'token':
                token_infos[col[1]] = TokenInfo.from_data(counts=series, num_nodes=count_df['count'], num_bins=num_bins)
            elif col[0] == 'number':
                number_infos[col[1]] = NumberInfo.from_data(
                    counts=series, values=data_df.loc[data_df['key'] == col[1], 'number'])
            elif col[0] == 'key':
                key_counts[col[1]] = ECDF.from_data(series, num_bins=num_bins)
        return super().from_data(token_infos=token_infos, number_infos=number_infos, key_counts=key_counts)

    @classmethod
    def reduce(cls, structs, weights=None, num_bins=None, processes=0, chunksize=1, disable=True):
        if weights is None:
            weights = np.full(len(structs), 1 / len(structs))
        else:
            weights = np.array(weights) / sum(weights)

        # 对齐
        number_infos = pd.DataFrame([struct.number_infos for struct in structs])
        token_infos = pd.DataFrame([struct.token_infos for struct in structs])
        key_counts = pd.DataFrame([struct.key_counts for struct in structs])

        # 填空值
        number_infos = number_infos.fillna(NumberInfo.from_data([0], []))
        token_infos = token_infos.fillna(TokenInfo.from_data(np.zeros(1), np.ones(1)))
        key_counts = key_counts.fillna(ECDF.from_data([0]))

        # reduce
        number_reducer = _NumberReducer(weights=weights, num_bins=num_bins)
        token_reducer = _TokenReducer(weights=weights, num_bins=num_bins)
        ecdf_reducer = _ECDFReducer(weights=weights, num_bins=num_bins)

        number_infos = {
            key: info for key, info in zip(
                number_infos,
                mp_run(number_reducer, [s for _, s in number_infos.iteritems()], processes=processes,
                       chunksize=chunksize, disable=disable, postfix='reduce number info')
            )
        }

        token_infos = {
            key: info for key, info in zip(
                token_infos,
                mp_run(token_reducer, [s for _, s in token_infos.iteritems()], processes=processes,
                       chunksize=chunksize, disable=disable, postfix='reduce token info')
            )
        }

        key_counts = {
            key: info for key, info in zip(
                key_counts,
                mp_run(ecdf_reducer, [s for _, s in key_counts.iteritems()], processes=processes,
                       chunksize=chunksize, disable=disable, postfix='reduce ecdf info')
            )
        }

        return super().reduce(structs, weights=weights,
                              token_infos=token_infos, number_infos=number_infos, key_counts=key_counts)

    def extra_repr(self) -> str:
        return 'num_keys={}, num_tokens={}, num_numbers={}'.format(self.num_keys, self.num_tokens, self.num_numbers)
