from typing import List, Dict

import dgl
import torch
import numpy as np
import pandas as pd


class EventGraph:
    SAMPLE_OBS_TIMESTAMP = 'sample_obs_timestamp'
    OBS_TIMESTAMP = 'obs_timestamp'
    EVENT_TIMESTAMP = 'event_timestamp'
    CENSOR_TIMESTAMP = 'censor_timestamp'
    
    DIFF_TIMESTAMP = 'diff_timestamp'
    ARRIVAL_TIME = 'arrival_time'
    CENSOR_TIME = 'censor_time'
    
    EVENT = 'event'
    SAMPLE_ID = 'sample_id'
    FEAT = 'feat'
    
    CAUSAL = 'causal'
    SURVIVAL = 'survival'
    
    NUM_EDGE_FEAT = 1
    
    def __init__(
        self,
        graph: dgl.DGLGraph,
        lookup_table: torch.Tensor,
        event_types: List[str],
    ):
        self.graph = graph.remove_self_loop(etype=self.SURVIVAL)
        self.lookup_table = lookup_table
        self.event_types = event_types
        
    @property
    def events(self):
        return self.graph.ndata[self.EVENT]
    
    @events.setter
    def events(self, event):
        self.graph.ndata[self.EVENT] = event
        
    @property
    def sample_ids(self):
        return self.graph.nodes[self.EVENT].data[self.SAMPLE_ID]
    
    @sample_ids.setter
    def sample_ids(self, sample_id):
        self.graph.nodes[self.EVENT].data[self.SAMPLE_ID] = sample_id
        
    @property
    def sample_obs_timestamps(self):
        return self.graph.nodes[self.EVENT].data[self.SAMPLE_OBS_TIMESTAMP]
    
    @sample_obs_timestamps.setter
    def sample_obs_timestamps(self, timestamps):
        self.graph.nodes[self.EVENT].data[self.SAMPLE_OBS_TIMESTAMP] = timestamps
        
    @property
    def obs_timestamps(self):
        return self.graph.nodes[self.EVENT].data[self.OBS_TIMESTAMP]
    
    @obs_timestamps.setter
    def obs_timestamps(self, timestamps):
        self.graph.nodes[self.EVENT].data[self.OBS_TIMESTAMP] = timestamps
        
    @property
    def event_timestamps(self):
        return self.graph.nodes[self.EVENT].data[self.EVENT_TIMESTAMP]
    
    @event_timestamps.setter
    def event_timestamps(self, timestamps):
        self.graph.nodes[self.EVENT].data[self.EVENT_TIMESTAMP] = timestamps
        
    @property
    def censor_timestamps(self):
        return self.graph.nodes[self.EVENT].data[self.CENSOR_TIMESTAMP]
    
    @censor_timestamps.setter
    def censor_timestamps(self, timestamps):
        self.graph.nodes[self.EVENT].data[self.CENSOR_TIMESTAMP] = timestamps
    
    @property
    def diff_timestamps(self):
        if self.DIFF_TIMESTAMP not in self.graph.edges[self.CAUSAL].data:
            self.graph.apply_edges(
                dgl.function.v_sub_u(self.EVENT_TIMESTAMP, self.EVENT_TIMESTAMP, self.DIFF_TIMESTAMP),
                etype=self.CAUSAL
            )
        return self.graph.edges[self.CAUSAL].data[self.DIFF_TIMESTAMP]
    
    @property
    def arrival_times(self):
        if self.ARRIVAL_TIME not in self.graph.nodes[self.EVENT].data:
            self.graph.update_all(
                dgl.function.v_sub_u(self.EVENT_TIMESTAMP, self.EVENT_TIMESTAMP, self.ARRIVAL_TIME),
                dgl.function.mean(self.ARRIVAL_TIME, self.ARRIVAL_TIME),
                etype=self.SURVIVAL
            )
        return self.graph.nodes[self.EVENT].data[self.ARRIVAL_TIME]
    
    @property
    def censor_times(self):
        if self.CENSOR_TIME not in self.graph.nodes[self.EVENT].data:
            self.graph.update_all(
                dgl.function.v_sub_u(self.CENSOR_TIMESTAMP, self.EVENT_TIMESTAMP, self.CENSOR_TIME),
                dgl.function.mean(self.CENSOR_TIME, self.CENSOR_TIME),
                etype=self.SURVIVAL
            )
        return self.graph.nodes[self.EVENT].data[self.CENSOR_TIME]
    
    @property
    def survival_times(self):
        censor_mask = ~torch.isnan(self.censor_timestamps)

        arrival_times = self.arrival_times.to(torch.float32).masked_fill(censor_mask, 0)
        censor_times = self.censor_times.to(torch.float32).masked_fill(~censor_mask, 0)

        lower_bounds = arrival_times + censor_times
        upper_bounds = arrival_times.masked_fill(censor_mask, np.inf)

        return lower_bounds, upper_bounds
        
    @property
    def edge_feats(self):
        if self.FEAT not in self.graph.edges[self.CAUSAL].data:
            feats = (self.diff_timestamps+1).log().unsqueeze(-1).to(torch.float32)
            feats = torch.masked_fill(feats, torch.isnan(feats), 0)
            self.graph.edges[self.CAUSAL].data[self.FEAT] = feats
        return self.graph.edges[self.CAUSAL].data[self.FEAT]
    
    @property
    def causal_graph(self):
        return self.graph.edge_type_subgraph(etypes=[self.CAUSAL])
    
    @property
    def survival_graph(self):
        return self.graph.edge_type_subgraph(etypes=[self.SURVIVAL])
        
    def to(self, *args, **kwargs):
        lookup_table = self.lookup_table.to(*args, **kwargs)
        graph = self.graph.to(*args, **kwargs)
        return self.__class__(
            graph=graph,
            lookup_table=lookup_table,
            event_types=self.event_types,
        )
    
    def pin_memory(self):
        self.lookup_table = self.lookup_table.pin_memory()
        self.graph = self.graph.pin_memory_()
        return self
        
    def to_simple(self):
        self.graph = self.graph.to_simple()


class EventGraphV2(EventGraph):
    ATTR = 'attr'
    
    def __init__(
        self,
        graph: dgl.DGLGraph,
        event_types: List[str],
    ):
        self.graph = graph.remove_self_loop(etype=self.SURVIVAL)
        self.event_types = event_types
        
    @property
    def events(self):
        if self.EVENT not in self.graph.nodes[self.EVENT].data:
            u, v = self.graph.edges(etype=self.ATTR)
            output = {}
            for i, j in zip(u.tolist(), v.tolist()):
                if j not in output:
                    output[j] = [i]
                else:
                    output[j].append(i)

            # 使用pandas对齐事件token长度
            output = pd.DataFrame([output[i] for i in range(len(output))])

            # 填充padding
            output = output.fillna(-1)

            # 在event末尾加上[cls]
            output['cls'] = self.graph.num_nodes(self.ATTR) - 1
            
            self.graph.nodes[self.EVENT].data[self.EVENT] = torch.tensor(
                output.values, dtype=torch.long, device=self.graph.device).contiguous()
        
        return self.graph.nodes[self.EVENT].data[self.EVENT]
        
    @property
    def lookup_table(self):
        return self.graph.nodes[self.ATTR].data[self.FEAT]
    
    @lookup_table.setter
    def lookup_table(self, lookup_table):
        self.graph.nodes[self.ATTR].data[self.FEAT] = lookup_table
        
    def to(self, *args, **kwargs):
        graph = self.graph.to(*args, **kwargs)
        return self.__class__(
            graph=graph,
            event_types=self.event_types,
        )
    
    def pin_memory(self):
        self.graph = self.graph.pin_memory_()
        return self
    
    def sample_subgraph(self, sample_ids):
        # 寻找事件点
        sub_event_nodes = np.nonzero(pd.Series(self.sample_ids.numpy()).isin(sample_ids).values)[0].tolist()

        # 寻找事件点对应的属性点
        attr_graph = self.graph.edge_type_subgraph([self.ATTR])
        sub_attr_graph = dgl.sampling.sample_neighbors(attr_graph, nodes={self.EVENT: sub_event_nodes}, fanout=-1)
        sub_attr_nodes = sub_attr_graph.edges()[0]
        sub_attr_nodes = sub_attr_nodes.unique().tolist()

        # 将[cls]对应的属性点加回去
        sub_attr_nodes.append(self.graph.num_nodes(self.ATTR)-1)

        # 采子图
        sub_graph = self.graph.subgraph({self.EVENT: sub_event_nodes, self.ATTR: sub_attr_nodes})
        sub_event_types = [self.event_types[i] for i in sub_event_nodes]
        return self.__class__(graph=sub_graph, event_types=sub_event_types)
    
    
class EventGraphV3(EventGraph):
    EVENT_FEAT = 'event_feat'
    
    def __init__(
        self,
        graph: dgl.DGLGraph,
        event_types: List[str],
        attributes: List[Dict[str, str]],
    ):
        self.graph = graph.remove_self_loop(etype=self.SURVIVAL)
        self.event_types = event_types
        self.attributes = attributes
        
    @property
    def events(self):
        if self.EVENT in self.graph.nodes[self.EVENT].data:
            return self.graph.nodes[self.EVENT].data[self.EVENT]
        
    @events.setter
    def events(self, events):
        self.graph.nodes[self.EVENT].data[self.EVENT] = events
        
    @property
    def event_feats(self):
        return self.graph.nodes[self.EVENT].data[self.EVENT_FEAT]
    
    @event_feats.setter
    def event_feats(self, event_feats):
        self.graph.nodes[self.EVENT].data[self.EVENT_FEAT] = event_feats
        
    def to(self, *args, **kwargs):
        graph = self.graph.to(*args, **kwargs)
        return self.__class__(
            graph=graph,
            event_types=self.event_types,
            attributes=self.attributes
        )
    
    def pin_memory(self):
        self.graph = self.graph.pin_memory_()
        return self
    
    def sample_subgraph(self, sample_ids):
        # 寻找事件点
        sub_event_nodes = np.nonzero(pd.Series(self.sample_ids.numpy()).isin(sample_ids).values)[0].tolist()

        # 采子图
        sub_graph = self.graph.subgraph({self.EVENT: sub_event_nodes})
        sub_event_types = [self.event_types[i] for i in sub_event_nodes]
        sub_attributes = [self.attributes[i] for i in sub_event_nodes]
        return self.__class__(graph=sub_graph, event_types=sub_event_types, attributes=sub_attributes)