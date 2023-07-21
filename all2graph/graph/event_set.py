import collections
from typing import Iterable, List

import numpy as np

from .event import Event
from ..utils import tokenizer, SEP, END


class EventSet:
    def __init__(self): 
        self.sample_ids = []
        self.obs_timestamps = []
        self.event_timestamps = []
        self.censor_timestamps = []
        self.event_types = []
        self.attributes = []
        self.foreign_keys = []
        
        self.sample_obs_timestamps = []
        
        self.causal_u = []
        self.causal_v = []
        
        self.survival_u = None
        self.survival_v = None
        
        self.lookup_table = None
        self.attr_edge_u = None
        self.attr_edge_v = None
        
    def __getitem__(self, i):
        return Event(
            obs_timestamp=self.obs_timestamps[i],
            event_timestamp=self.event_timestamps[i],
            censor_timestamp=self.censor_timestamps[i],
            event_type=self.event_types[i],
            attributes=self.attributes[i],
            foreign_keys=self.foreign_keys[i],
        )
    
    def add_causal_edges(self, n, src_nodes, dst_node):
        # 处理时间戳相同的情况
        i = 0
        prev_timestamp = dst_timestamp = self.event_timestamps[dst_node]
        for src in src_nodes:
            timestamp = self.event_timestamps[src]
            if timestamp >= dst_timestamp:
                continue
            if i < n or timestamp == prev_timestamp:
                self.causal_u.append(src)
                self.causal_v.append(dst_node)
                prev_timestamp = timestamp
                i += 1
            else:
                break
        
    def add_events(
        self,
        events: Iterable[Event],
        sample_id: int,
        sample_obs_timestamp: int,
        degree: int,
    ):
        # 按照event_timestamp或censor_timestamp中非空的那个排序
        def sorted_key(event: Event):
            return event.event_timestamp or np.inf, event.obs_timestamp or -np.inf
            
        events = sorted(events, key=sorted_key)
        
        # 用户添加因果边的缓存
        src_by_event_type = {}
        src_by_foreign_key = {}
        
        for current_node, event in enumerate(events, start=len(self.obs_timestamps)):
            self.event_timestamps.append(event.event_timestamp or np.nan)
            
            # self loop
            self.causal_u.append(current_node)
            self.causal_v.append(current_node)
            
            # 执行顺序不能乱改!!!!!
            # 添加因果边
            if event.event_timestamp is not None and (event.obs_timestamp is None or event.event_timestamp < sample_obs_timestamp):
                # 按照事件类型遍历已发生的事件
                # 将最近的n个事件连上边
                for src_type, src_nodes in src_by_event_type.items():
                    if event.event_type == src_type:
                        self.add_causal_edges(degree, src_nodes, current_node)
                    else:
                        self.add_causal_edges(1, src_nodes, current_node)
                
                # 插入到当前类型的已发生事件中
                if event.event_type not in src_by_event_type:
                    src_by_event_type[event.event_type] = collections.deque()
                src_by_event_type[event.event_type].appendleft(current_node)

                # 根据外键添加边
                for key, value in event.foreign_keys.items():
                    if (key, value) in src_by_foreign_key:
                        src_nodes = src_by_foreign_key[key, value]
                        # 插入新的边
                        self.add_causal_edges(degree, src_nodes, current_node)
                    else:
                        src_by_foreign_key[key, value] = collections.deque()
                    # 插入新的事件
                    src_by_foreign_key[key, value].appendleft(current_node)
        
        self.censor_timestamps.extend(event.censor_timestamp or np.nan for event in events)
        self.event_types.extend(event.event_type for event in events)
        self.attributes.extend(event.attributes for event in events)
        self.foreign_keys.extend(event.foreign_keys for event in events)
        self.obs_timestamps.extend(event.obs_timestamp or np.nan for event in events)
        self.sample_ids.extend([sample_id]*len(events))
        self.sample_obs_timestamps.extend([sample_obs_timestamp]*len(events))
        
    def gen_survival_edges(self):
        if self.survival_u is None or self.survival_v is None:
            u = []
            v = []
            src_by_sample_and_timestamp = {}
            for i, (sample_id, event_timestamp) in enumerate(zip(self.sample_ids, self.event_timestamps)):
                if np.isnan(event_timestamp):
                    continue
                if (sample_id, event_timestamp) not in src_by_sample_and_timestamp:
                    src_by_sample_and_timestamp[sample_id, event_timestamp] = []
                src_by_sample_and_timestamp[sample_id, event_timestamp].append(i)

            for i, (sample_id, obs_timestamp) in enumerate(zip(self.sample_ids, self.obs_timestamps)):
                if (sample_id, obs_timestamp) in src_by_sample_and_timestamp:
                    temp = src_by_sample_and_timestamp[sample_id, obs_timestamp]
                    u.extend(temp)
                    v.extend([i]*len(temp))
            self.survival_u = u
            self.survival_v = v
        return self.survival_u, self.survival_v
        
    @property
    def num_nodes(self):
        return len(self.sample_ids)
    
    @property
    def num_causal_edges(self):
        return len(self.causal_u)
    
    @property
    def unique_event_types(self):
        return set(self.event_types)
    
    @property
    def num_event_types(self):
        return len(self.unique_event_types)
    
    @property
    def unique_tokens(self):
        strs = set()
        
        def add_str(s):
            if isinstance(s, str):
                strs.add(s)
        
        for event_type in self.event_types:
            add_str(event_type)
            
        for attr in self.attributes:
            for key, value in attr.items():
                add_str(key)
                add_str(value)

        tokens = set()
        for s in strs:
            tokens.update(tokenizer.lcut(s))
                
        return tokens
    
    @property
    def tokens(self):
        # 缓存字符串的分词
        cache = {}

        def lcut_wrap(s) -> List[str]:
            if s not in cache:
                if isinstance(s, type):
                    cache[s] = [s]
                elif isinstance(s, (int, float)):
                    cache[s] = list(str(s))
                else:    
                    cache[s] = tokenizer.lcut(str(s))
            return cache[s]

        # 每一行表示一个type或key-value的tokens
        lookup_table = []

        # type或key-value在lookup_table中的坐标
        lookup_idx = {}

        # 每个事件包含的type和attr在lookup_table中的坐标
        event_tokens = []

        for event_type, attributes in zip(self.event_types, self.attributes):
            temp_idx = []

            if event_type not in lookup_idx:
                lookup_idx[event_type] = len(lookup_table)
                lookup_table.append(lcut_wrap(event_type))
            temp_idx.append(lookup_idx[event_type])
            
            for key, value in attributes.items():
                if (key, value) not in lookup_idx:
                    lookup_idx[key, value] = len(lookup_table)
                    lookup_table.append(lcut_wrap(key)+[SEP]+lcut_wrap(value)+[END])
                temp_idx.append(lookup_idx[key, value])

            event_tokens.append(temp_idx)
            
        return lookup_table, event_tokens
    
    def gen_attribute_lookup_table(self):
        if self.lookup_table is None or self.attr_edge_u is None or self.attr_edge_v is None:
            # 每一行表示一个type或key-value的tokens
            lookup_table = []

            # type或key-value在lookup_table中的坐标
            lookup_idx = {}

            # 属性边
            attr_edge_u = []
            attr_edge_v = []

            for event_i, (event_type, attributes) in enumerate(zip(self.event_types, self.attributes)):
                if event_type not in lookup_idx:
                    lookup_idx[event_type] = len(lookup_table)
                    lookup_table.append([event_type, ''])
                attr_edge_u.append(lookup_idx[event_type])
                attr_edge_v.append(event_i)

                for key, value in attributes.items():
                    value = str(value)
                    if (key, value) not in lookup_idx:
                        lookup_idx[key, value] = len(lookup_table)
                        lookup_table.append([key, value])
                    attr_edge_u.append(lookup_idx[key, value])
                    attr_edge_v.append(event_i)

            self.lookup_table = lookup_table
            self.attr_edge_u = attr_edge_u
            self.attr_edge_v = attr_edge_v
        return self.lookup_table, self.attr_edge_u, self.attr_edge_v
        
    def __repr__(self):
        return f'{self.__class__.__name__}(num_event_types={self.num_event_types}, num_nodes={self.num_nodes}, num_causal_edges={self.num_causal_edges})'

