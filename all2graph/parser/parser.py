import json
import string
import os
from copy import deepcopy
from typing import Iterable, List
from itertools import combinations
from gzip import GzipFile

import numpy as np
import dgl
import torch
import pandas as pd

from ..graph import (
    EventGraph,
    EventSet,
    Event,
    EventGraphV2,
    EventGraphV3
)
from ..utils import (
    SpecialToken,
    CLS,
    UNKNOWN,
    END,
    PADDING,
    SEP,
    iter_csv,
    mp_run
)

class BaseParser:
    SPECIAL_TOKENS = 'special_tokens'
    COMMON_TOKENS = 'common_tokens'

    def __init__(
        self,
        degree: int,
        num_layers: int,
        target_cols: Iterable[str],
        event_cols: Iterable[str],
        obs_timestamp_key=EventGraph.OBS_TIMESTAMP,
        event_timestamp_key=EventGraph.EVENT_TIMESTAMP,
        censor_timestamp_key=EventGraph.CENSOR_TIMESTAMP,
        foreign_keys: Iterable[str]=None,
        ignored_keys: Iterable[str]=None,
    ):
        assert isinstance(degree, int)
        assert isinstance(num_layers, int)
        assert isinstance(obs_timestamp_key, str)
        assert isinstance(event_timestamp_key, str)
        assert isinstance(censor_timestamp_key, str)
        
        self.degree = degree
        self.num_layers = num_layers
        self.obs_timestamp_key = obs_timestamp_key
        self.event_timestamp_key = event_timestamp_key
        self.censor_timestamp_key = censor_timestamp_key
        
        self.target_cols = set(target_cols)
        self.event_cols = set(event_cols)
        self.foreign_keys = set(foreign_keys or set())
        self.ignored_keys = set(ignored_keys or set())
        
        def check_type(inputs):
            assert isinstance(inputs, set)
            for x in inputs:
                assert isinstance(x, str)
        
        check_type(self.target_cols)
        check_type(self.event_cols)
        check_type(self.foreign_keys)
        check_type(self.ignored_keys)
        
        def check_disjoint(*sets):
            for a, b in combinations(sets, 2):
                assert a.isdisjoint(b)
                
        check_disjoint(
            self.target_cols,
            self.event_cols
        )
       
        check_disjoint(
            {self.obs_timestamp_key},
            {self.event_timestamp_key},
            {self.censor_timestamp_key},
            self.foreign_keys,
            self.ignored_keys
        )
        
    def parse_df(self, df: pd.DataFrame) -> EventSet:
        
        def gen_event(event_type: str, event_data: dict) -> Event:
            obs_timestamp = event_data.pop(self.obs_timestamp_key, None)
            event_timestamp = event_data.pop(self.event_timestamp_key, None)
            censor_timestamp = event_data.pop(self.censor_timestamp_key, None)
            
            if event_timestamp is None and censor_timestamp is None and obs_timestamp is None:
                return

            for key in self.ignored_keys:
                event_data.pop(key, None)

            foreign_keys = {}
            for key in self.foreign_keys:
                if key in event_data:
                    foreign_keys[key] = event_data.pop(key)

            return Event(
                obs_timestamp=obs_timestamp,
                event_timestamp=event_timestamp,
                censor_timestamp=censor_timestamp,
                event_type=event_type,
                attributes=event_data,
                foreign_keys=foreign_keys
            )
        
        def parse_row() -> Iterable[Event]:
            for col in self.event_cols:
                json_str = getattr(row, col, None)
                if isinstance(json_str, str):
                    json_objs = json.loads(json_str)
                elif isinstance(json_str, (list, dict)):
                    json_objs = deepcopy(json_str)
                else:
                    continue

                if isinstance(json_objs, dict):
                    event = gen_event(col, json_objs)
                    if event is not None:
                        yield event
                    continue
                
                for json_obj in json_objs:
                    event = gen_event(col, json_obj)
                    if event is not None:
                        yield event
        
        graph = EventSet()
        for i, row in enumerate(df.itertuples()):
            graph.add_events(
                parse_row(),
                sample_id=i,
                sample_obs_timestamp=getattr(row, self.obs_timestamp_key),
                degree=self.degree,
            )
        return graph
    
    def __call__(self, df: pd.DataFrame) -> EventGraph:
        event_graph = self.parse_df(df)
        return self.gen_event_graph(event_graph)
    
    def get_targets(self, df: pd.DataFrame):        
        output = {}
        for k in self.target_cols:
            if k in df:
                output[k] = torch.tensor(df[k].values, dtype=torch.float32) 
            else:
                output[k] = torch.full((df.shape[0], ), fill_value=np.nan, dtype=torch.float32)
        return output
    
    def generate(self, df, pre_func=None):
        if pre_func is not None:
            df = pre_func(df)
        parsed = self(df)
        df = df.drop(columns=list(self.event_cols))
        for col, value in self.get_targets(df).items():
            df[col] = value.numpy()
        return parsed, df
    
    def generator(self, src, chunksize, processes=0, disable=False, unordered=False, pre_func=None, **kwargs):
        data = iter_csv(src, chunksize=chunksize, **kwargs)
        for output in mp_run(
            self.generate,
            data,
            kwds={'pre_func': pre_func},
            processes=processes,
            disable=disable,
            unordered=unordered
        ):
            yield output
            
    def _save_graph(self, inputs, dst, compresslevel=6, pre_func=None, path_col='path'):
        i, df = inputs
        if pre_func is not None:
            df = pre_func(df)
        graph = self(df)
        df = df.drop(columns=list(self.event_cols))
        path = os.path.join(dst, f'{i}.all2graph.{graph.__class__.__name__}.gzip')
        df[path_col] = path
        with GzipFile(path, 'wb', compresslevel=compresslevel) as myzip:
            label = self.get_targets(df)
            torch.save([graph, label], myzip)
        return df
        
    def save_graph(self, src, dst, chunksize, processes=0, disable=False, unordered=False, pre_func=None, index=False, path_col='path', compresslevel=6, **kwargs):
        # 检查文件夹是否存在，如果存在则报错，否则创建文件夹
        if os.path.exists(dst):
            raise ValueError('{} already exists'.format(dst))
        os.mkdir(dst)
        
        meta_df = []
        for df in mp_run(
            self._save_graph,
            enumerate(iter_csv(src, chunksize=chunksize, **kwargs)),
            kwds={'dst': dst, 'compresslevel': compresslevel, 'pre_func': pre_func, 'path_col': path_col},
            processes=processes,
            disable=disable,
            unordered=unordered
        ):
            meta_df.append(df)
        
        meta_df = pd.concat(meta_df)
        meta_df.to_csv(f'{dst}_path.zip', index=index)


class Parser(BaseParser):
    def __init__(
        self,
        degree: int,
        num_layers: int,
        target_cols: Iterable[str],
        event_cols: Iterable[str],
        obs_timestamp_key=EventGraph.OBS_TIMESTAMP,
        event_timestamp_key=EventGraph.EVENT_TIMESTAMP,
        censor_timestamp_key=EventGraph.CENSOR_TIMESTAMP,
        foreign_keys: Iterable[str]=None,
        ignored_keys: Iterable[str]=None,
    ):
        super().__init__(
            degree=degree,
            num_layers=num_layers,
            target_cols=target_cols,
            event_cols=event_cols,
            obs_timestamp_key=obs_timestamp_key,
            event_timestamp_key=event_timestamp_key,
            censor_timestamp_key=censor_timestamp_key,
            foreign_keys=foreign_keys,
            ignored_keys=ignored_keys,
        )
        
        self._dictionary = {}
        self.add_tokens([PADDING, UNKNOWN, END, CLS, SEP, str(None), str(True), str(False)])
        self.add_tokens(string.printable)
        
    def add_tokens(self, tokens):
        for token in tokens:
            assert isinstance(token, str) or issubclass(token, SpecialToken)
            if token not in self._dictionary:
                self._dictionary[token] = len(self._dictionary)
                
    def dump_dictionary(self, path=None):
        output = {}
        output[self.SPECIAL_TOKENS] = {k.__name__: v for k, v in self._dictionary.items() if isinstance(k, type) and issubclass(k, SpecialToken)}
        output[self.COMMON_TOKENS] = {k: v for k, v in self._dictionary.items() if not isinstance(k, type) or not issubclass(k, SpecialToken)}
        if path is None:
            return output
        else:
            with open(path, 'w') as file:
                json.dump(output, file)

    def load_dictionary(self, path):
        with open(path, 'r') as file:
            tokens = json.load(file)
        new_dictionary = {}
        for k, v in tokens[self.SPECIAL_TOKENS].items():
            new_dictionary[globals()[k]] = v
        for k, v in tokens[self.COMMON_TOKENS].items():
            new_dictionary[k] = v
        assert set(new_dictionary.values()) == set(range(len(new_dictionary)))
        self._dictionary = new_dictionary
    
    def find_all_tokens(self, df):
        graph = self.parse_df(df)
        return graph.unique_tokens
    
    def gen_dictionary(
        self,
        inputs,
        chunksize,
        processes=0,
        disable=False,
        pre_func=None,
        **kwargs
    ):
        data = iter_csv(inputs, chunksize=chunksize, pre_func=pre_func, **kwargs)
        for tokens in mp_run(self.find_all_tokens, data, processes=processes, disable=disable):
            self.add_tokens(tokens)
        
    @property
    def num_embeddings(self):
        return len(self._dictionary)
    
    @property
    def cls(self):
        return self._dictionary[CLS]
    
    @property
    def padding(self):
        return self._dictionary[PADDING]
    
    @property
    def unknown(self):
        return self._dictionary[UNKNOWN]
    
    @property
    def end(self):
        return self._dictionary[END]
    
    @property
    def true(self):
        return self._dictionary[str(True)]
    
    @property
    def false(self):
        return self._dictionary[str(False)]
        
    def vectorization(self, tokens: List[List[str]]) -> torch.Tensor:
        unknown = self.unknown
        output = pd.DataFrame(tokens)
        output = output.fillna(PADDING)
        output = output.applymap(lambda x: self._dictionary.get(x, unknown))
        assert CLS.__name__ not in output
        output[CLS.__name__] = self.cls
        return torch.tensor(output.values, dtype=torch.long).contiguous()
    
    def tensor2str(self, tensor: torch.Tensor) -> List[str]:
        reverse_dict = {}
        for k, v in self._dictionary.items():
            if isinstance(k, type):
                reverse_dict[v] = k.__name__
            else:
                reverse_dict[v] = k
        output = pd.DataFrame(tensor.numpy())
        output = output.applymap(lambda x: reverse_dict[x])
        output = output.apply(lambda x: ' '.join(x), axis=1).tolist()
        return output
        
    def gen_event_graph(self, event_set: EventSet) -> EventGraph:
        # 构造图        
        graph = dgl.heterograph(
            {
                (EventGraph.EVENT, EventGraph.SURVIVAL, EventGraph.EVENT): event_set.gen_survival_edges(),
                (EventGraph.EVENT, EventGraph.CAUSAL, EventGraph.EVENT): (event_set.causal_u, event_set.causal_v)
            },
            num_nodes_dict={EventGraph.EVENT: event_set.num_nodes}
        )
        
        # 处理event tokens
        lookup_table, event_tokens = event_set.tokens
        
        # 在lookup_table中添加[cls]
        lookup_table = self.vectorization(lookup_table + [[CLS]])
        
        # 使用pandas对齐事件token长度
        event_tokens = pd.DataFrame(event_tokens)
        
        # 填充padding
        event_tokens = event_tokens.fillna(-1)
        
        # 在event末尾加上[cls]
        assert CLS.__name__ not in event_tokens
        event_tokens[CLS.__name__] = lookup_table.shape[0] - 1
        
        # 封装
        output = EventGraph(
            graph,
            lookup_table=lookup_table,
            event_types=event_set.event_types,
        )
        output.events = torch.tensor(event_tokens.values, dtype=torch.long).contiguous()
        output.sample_ids = torch.tensor(event_set.sample_ids, dtype=torch.long)
        output.sample_obs_timestamps = torch.tensor(event_set.sample_obs_timestamps, dtype=torch.float64)
        output.obs_timestamps = torch.tensor(event_set.obs_timestamps, dtype=torch.float64)
        output.event_timestamps = torch.tensor(event_set.event_timestamps, dtype=torch.float64)
        output.censor_timestamps = torch.tensor(event_set.censor_timestamps, dtype=torch.float64)
        
        output.to_simple()
        return output
    
    
class ParserV2(BaseParser):
    def __init__(self, *args, transformers_tokenizer, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformers_tokenizer = transformers_tokenizer
        del self._dictionary
        
    @property
    def num_embeddings(self):
        return self.transformers_tokenizer.vocab_size
    
    @property
    def padding(self):
        return self.transformers_tokenizer.pad_token_id
    
    def gen_event_graph(self, event_set: EventSet) -> EventGraph:
        lookup_table, attr_edge_u, attr_edge_v = event_set.gen_attribute_lookup_table()
        # 在lookup_table中添加[cls]
        lookup_table.append([self.transformers_tokenizer.cls_token, ''])
        
        # encode lookup_table
        lookup_table = self.transformers_tokenizer(
            lookup_table,
            add_special_tokens=True,
            padding=True,
            return_tensors='pt',
            return_token_type_ids=False,
            return_attention_mask=False,
            return_special_tokens_mask=False
        )['input_ids']
        
        # 将cls放到末尾
        lookup_table = torch.cat([lookup_table[:, 1:], lookup_table[:, [0]]], dim=1).contiguous()
        
        # 构造图        
        graph = dgl.heterograph(
            {
                (EventGraphV2.EVENT, EventGraphV2.SURVIVAL, EventGraphV2.EVENT): event_set.gen_survival_edges(),
                (EventGraphV2.EVENT, EventGraphV2.CAUSAL, EventGraphV2.EVENT): (event_set.causal_u, event_set.causal_v),
                (EventGraphV2.ATTR, EventGraphV2.ATTR, EventGraphV2.EVENT): (attr_edge_u, attr_edge_v),
            },
            num_nodes_dict={
                EventGraphV2.EVENT: event_set.num_nodes,
                EventGraphV2.ATTR: lookup_table.shape[0]
            }
        )
        
        # 封装
        output = EventGraphV2(
            graph,
            event_types=event_set.event_types,
        )
        output.lookup_table = lookup_table
        output.sample_ids = torch.tensor(event_set.sample_ids, dtype=torch.long)
        output.sample_obs_timestamps = torch.tensor(event_set.sample_obs_timestamps, dtype=torch.float64)
        output.obs_timestamps = torch.tensor(event_set.obs_timestamps, dtype=torch.float64)
        output.event_timestamps = torch.tensor(event_set.event_timestamps, dtype=torch.float64)
        output.censor_timestamps = torch.tensor(event_set.censor_timestamps, dtype=torch.float64)
        
        output.to_simple()
        return output
    

class ParserV3(BaseParser):
    def gen_event_graph(self, event_set: EventSet) -> EventGraph:
        # 构造图        
        graph = dgl.heterograph(
            {
                (EventGraphV2.EVENT, EventGraphV2.SURVIVAL, EventGraphV2.EVENT): event_set.gen_survival_edges(),
                (EventGraphV2.EVENT, EventGraphV2.CAUSAL, EventGraphV2.EVENT): (event_set.causal_u, event_set.causal_v),
            },
            num_nodes_dict={
                EventGraphV2.EVENT: event_set.num_nodes,
            }
        )
        
        # 封装
        output = EventGraphV3(
            graph,
            event_types=event_set.event_types,
            attributes=event_set.attributes,
        )
        output.sample_ids = torch.tensor(event_set.sample_ids, dtype=torch.long)
        output.sample_obs_timestamps = torch.tensor(event_set.sample_obs_timestamps, dtype=torch.float64)
        output.obs_timestamps = torch.tensor(event_set.obs_timestamps, dtype=torch.float64)
        output.event_timestamps = torch.tensor(event_set.event_timestamps, dtype=torch.float64)
        output.censor_timestamps = torch.tensor(event_set.censor_timestamps, dtype=torch.float64)
        
        output.to_simple()
        return output