import numpy as np
import pandas as pd

from ..macro import NULL
from ..meta_graph import MetaGraph


class Transformer:
    def __init__(self, meta_graph: MetaGraph, min_tf=0, min_df=0, min_tf_idf=0, top_tf=None, top_tf_idf=None,
                 lower=0.05, upper=0.05, split_name=True):
        self.number_range = {
            key: (ecdf.meta_data.get_quantiles(lower), ecdf.meta_data.get_quantiles(upper))
            for key, ecdf in meta_graph.meta_numbers.items()
            if ecdf.meta_data.mean_var[1] > 0
        }

        self.split_name = split_name
        self.string_mapper = {}
        for name in meta_graph.meta_name:
            if split_name:
                import jieba
                for key in jieba.cut(name):
                    if key not in self.string_mapper:
                        self.string_mapper[key] = len(self.string_mapper)
            elif name not in self.string_mapper:
                self.string_mapper[name] = len(self.string_mapper)

        tf_dict = {}
        tf_idf_dict = {}
        mean_num_nodes = meta_graph.meta_string.freq.mean
        for key, ecdf in meta_graph.meta_string.items():
            tf = ecdf.mean * mean_num_nodes
            df = ecdf.get_probs(0)
            tf_idf = tf / np.log(1 / (1 + df))
            print(tf, df, tf_idf)
            if tf >= min_tf and df >= min_df and tf_idf >= min_tf_idf:
                tf_dict[key] = tf
                tf_idf_dict[key] = tf_idf

        if top_tf is not None:
            tf_dict = pd.Series(tf_dict)
            tf_dict = tf_dict.sort_values(ascending=False)
            tf_dict = tf_dict.iloc[:top_tf].to_dict()

        if top_tf_idf is not None:
            tf_idf_dict = pd.Series(tf_idf_dict)[tf_dict]
            tf_idf_dict = tf_idf_dict.sort_values(ascending=False)
            tf_idf_dict = tf_idf_dict.iloc[:top_tf_idf].to_dict()

        for key in meta_graph.meta_string:
            if key in tf_dict and key in tf_idf_dict and key not in self.string_mapper:
                self.string_mapper[key] = len(self.string_mapper)

        if NULL not in self.string_mapper:
            self.string_mapper[NULL] = len(self.string_mapper)

        assert NULL in self.string_mapper
        assert len(self.string_mapper) == len(set(self.string_mapper.values())), (
                len(self.string_mapper), len(set(self.string_mapper.values()))
        )
        assert len(self.string_mapper) == max(list(self.string_mapper.values())) + 1

