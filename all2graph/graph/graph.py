from typing import List, Union

import dgl
import numpy as np
import torch

from .raw import RawGraph
from ..globals import EPSILON, EDGE


class Graph:
    def __init__(self, graph: dgl.DGLHeteroGraph):
        self.graph = graph

    @classmethod
    def from_raw_graph(
            cls,
            meta_graph: RawGraph,
            value_graph: RawGraph,
            meta_key: Union[np.ndarray, List[int]],
            meta_value: Union[np.ndarray, List[int]],
            meta_symbol: Union[np.ndarray, List[int]],
            meta_component_id: Union[np.ndarray, List[int]],
            meta_edge_key: Union[np.ndarray, List[int]],
            value: Union[np.ndarray, List[int]],
            number: Union[np.ndarray, List[float]],
            symbol: Union[np.ndarray, List[int]],
            meta_node_id: Union[np.ndarray, List[int]],
    ):
        graph = dgl.heterograph(
            data_dict={
                (META, EDGE, META): (
                    torch.tensor(meta_graph.src, dtype=torch.long), torch.tensor(meta_graph.dst, dtype=torch.long)),
                (VALUE, EDGE, VALUE): (
                    torch.tensor(value_graph.src, dtype=torch.long), torch.tensor(value_graph.dst, dtype=torch.long)),
                (VALUE, EDGE, META): (
                    torch.arange(value_graph.num_nodes, dtype=torch.long), torch.tensor(meta_node_id, dtype=torch.long))
            },
            num_nodes_dict={META: meta_graph.num_nodes, VALUE: value_graph.num_nodes}
        )
        # 元图属性
        graph.nodes[META].data[KEY] = torch.tensor(meta_key, dtype=torch.long)
        graph.nodes[META].data[VALUE] = torch.tensor(meta_value, dtype=torch.long)
        graph.nodes[META].data['symbol'] = torch.tensor(meta_symbol, dtype=torch.long)
        graph.nodes[META].data['component_id'] = torch.tensor(meta_component_id, dtype=torch.long)
        graph.edges[(META, EDGE, META)].data[KEY] = torch.tensor(meta_edge_key, dtype=torch.long)

        # 值图属性
        graph.nodes[VALUE].data[NUMBER] = torch.tensor(number, dtype=torch.float32)
        graph.nodes[VALUE].data[VALUE] = torch.tensor(value, dtype=torch.long)
        graph.nodes[VALUE].data['symbol'] = torch.tensor(symbol, dtype=torch.long)
        return cls(graph)

    def __eq__(self, other, debug=False):
        if (self.meta_graph.edges()[0] != other.meta_graph.edges()[0]).any():
            if debug:
                print('meta_graph.src not equal')
            return False
        if (self.meta_graph.edges()[1] != other.meta_graph.edges()[1]).any():
            if debug:
                print('meta_graph.dst not equal')
            return False
        if (self.value_graph.edges()[0] != other.value_graph.edges()[0]).any():
            if debug:
                print('meta_graph.src not equal')
            return False
        if (self.value_graph.edges()[1] != other.value_graph.edges()[1]).any():
            if debug:
                print('meta_graph.dst not equal')
            return False
        if (self.meta_value != other.meta_value).any():
            if debug:
                print('meta_value not equal')
            return False
        if (self.meta_key != other.meta_key).any():
            if debug:
                print('meta_key not equal')
            return False
        if (self.meta_symbol != other.meta_symbol).any():
            if debug:
                print('meta_symbol not equal')
            return False
        if (self.meta_component_id != other.meta_component_id).any():
            if debug:
                print('meta_component_id not equal')
            return False
        if (self.value != other.value).any():
            if debug:
                print('meta_component_id not equal')
            return False
        if (self.number - other.number).abs().max() < EPSILON:
            if debug:
                print('number not equal')
            return False
        if (self.symbol != other.symbol).any():
            if debug:
                print('symbol not equal')
            return False
        if (self.meta_node_id != other.meta_node_id).any():
            if debug:
                print('meta_node_id not equal')
            return False
        if (self.meta_edge_id != other.meta_edge_id).any():
            if debug:
                print('meta_edge_id not equal')
            return False
        return True

    def __repr__(self):
        return self.graph.__repr__()

    @property
    def meta_graph(self):
        meta_graph = dgl.node_type_subgraph(self.graph, [META])
        return dgl.to_homogeneous(meta_graph)

    @property
    def value_graph(self):
        value_graph = dgl.node_type_subgraph(self.graph, [VALUE])
        return dgl.to_homogeneous(value_graph)

    @property
    def number(self):
        return self.graph.nodes[VALUE].data[NUMBER]

    @property
    def value(self):
        return self.graph.nodes[VALUE].data[VALUE]

    @property
    def symbol(self):
        return self.graph.nodes[VALUE].data['symbol']

    @property
    def meta_node_id(self):
        return self.graph.edges(etype=(VALUE, EDGE, META))[1]

    @property
    def meta_edge_id(self):
        src, dst = self.graph.edges(etype=(VALUE, EDGE, VALUE))
        meta_node_id = self.meta_node_id
        meta_scr = meta_node_id[src]
        meta_dst = meta_node_id[dst]
        return self.graph.edge_ids(meta_scr, meta_dst, etype=(META, EDGE, META))

    @property
    def meta_symbol(self):
        return self.graph.nodes[META].data['symbol']

    @property
    def meta_component_id(self):
        return self.graph.nodes[META].data['component_id']

    @property
    def meta_value(self):
        return self.graph.nodes[META].data[VALUE]

    @property
    def meta_key(self):
        return self.graph.nodes[META].data[KEY]

    @property
    def key(self):
        return self.meta_key[self.meta_node_id]

    @property
    def component_id(self):
        return self.meta_component_id[self.meta_node_id]

    @property
    def meta_edge_key(self):
        return self.graph.edges[(META, EDGE, META)].data[KEY]

    @property
    def edge_key(self):
        return self.meta_edge_key[self.meta_edge_id]

    def target_mask(self, target_symbol: list):
        target_symbol = torch.tensor(target_symbol, dtype=torch.long, device=self.symbol.device)
        return (self.symbol.view(-1, 1) == target_symbol).any(-1)

    def save(self, filename, labels=None):
        dgl.save_graphs(filename, [self.graph], labels=labels)

    @classmethod
    def load(cls, filename):
        graphs, labels = dgl.load_graphs(filename)
        return cls(graph=graphs[0]), labels

    def to(self, *args, **kwargs):
        self.graph = self.graph.to(*args, **kwargs)
        return self

    def to_df(self, *attrs):
        """
        转换成一个可读性强的dataframe
        Args:
            *attrs: 属性，如果key，value，number等

        Returns:

        """
        import pandas as pd

        src, dst = self.value_graph.edges()
        src = src.detach().cpu().numpy()
        dst = dst.detach().cpu().numpy()

        output = {
            'src': src,
            'dst': dst,
        }
        for attr in attrs:
            output['src_{}'.format(attr)] = getattr(self, attr)[src].detach().cpu().numpy()

        for attr in attrs:
            output['dst_{}'.format(attr)] = getattr(self, attr)[dst].detach().cpu().numpy()
        return pd.DataFrame(output)

    def pin_memory(self):
        for k, v in self.graph.ndata.items():
            for kk, vv in v.items():
                self.graph.ndata[k][kk] = vv.pin_memory()

        for k, v in self.graph.edata.items():
            for kk, vv in v.items():
                self.graph.edata[k][kk] = vv.pin_memory()

        return self

    def component_subgraph(self, i):
        return Graph(dgl.node_subgraph(self.graph, {META: self.meta_component_id == i, VALUE: self.component_id == i}))

    @classmethod
    def batch(cls, graphs):
        return cls(dgl.batch([graph.graph for graph in graphs]))
