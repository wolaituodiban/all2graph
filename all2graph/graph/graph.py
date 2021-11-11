from typing import Union

import dgl
import torch

from .raw import RawGraph
from ..globals import EPSILON
from ..version import __version__
from ..preserves import KEY, VALUE, NUMBER


class Graph:
    def __init__(self, meta_graph: Union[RawGraph, dgl.DGLGraph], graph: Union[RawGraph, dgl.DGLGraph], meta_key,
                 meta_value, meta_symbol, meta_component_id, meta_edge_key, value, number, symbol, meta_node_id,
                 meta_edge_id):
        self.version = __version__
        # 构建元图
        if isinstance(meta_graph, RawGraph):
            self.meta_graph: dgl.DGLGraph = dgl.graph(
                data=(torch.tensor(meta_graph.src, dtype=torch.long), torch.tensor(meta_graph.dst, dtype=torch.long)),
                num_nodes=meta_graph.num_nodes
            )
        elif isinstance(meta_graph, dgl.DGLGraph):
            self.meta_graph = meta_graph
        else:
            raise TypeError('unknown type of meta_graph: {}'.format(type(meta_graph)))

        self.meta_graph.ndata[KEY] = meta_key if isinstance(meta_key, torch.Tensor) \
            else torch.tensor(meta_key, dtype=torch.long)
        self.meta_graph.ndata[VALUE] = meta_value if isinstance(meta_value, torch.Tensor) \
            else torch.tensor(meta_value, dtype=torch.long)
        self.meta_graph.ndata['meta_symbol'] = meta_symbol if isinstance(meta_symbol, torch.Tensor) \
            else torch.tensor(meta_symbol, dtype=torch.long)
        self.meta_graph.ndata['meta_component_id'] = meta_component_id if isinstance(meta_component_id, torch.Tensor) \
            else torch.tensor(meta_component_id, dtype=torch.long)

        self.meta_graph.edata[KEY] = meta_edge_key if isinstance(meta_edge_key, torch.Tensor) \
            else torch.tensor(meta_edge_key, dtype=torch.long)

        # 构建值图
        if isinstance(graph, RawGraph):
            self.value_graph: dgl.DGLGraph = dgl.graph(
                data=(torch.tensor(graph.src, dtype=torch.long), torch.tensor(graph.dst, dtype=torch.long)),
                num_nodes=graph.num_nodes
            )
        elif isinstance(graph, dgl.DGLGraph):
            self.value_graph = graph
        else:
            raise TypeError('unknown type of value_graph: {}'.format(type(graph)))

        self.value_graph.ndata[NUMBER] = number if isinstance(symbol, torch.Tensor) else \
            torch.tensor(number, dtype=torch.float32)
        self.value_graph.ndata[VALUE] = value if isinstance(symbol, torch.Tensor) else \
            torch.tensor(value, dtype=torch.long)
        self.value_graph.ndata['symbol'] = symbol if isinstance(symbol, torch.Tensor) else \
            torch.tensor(symbol, dtype=torch.long)
        self.value_graph.ndata['meta_node_id'] = meta_node_id if isinstance(meta_node_id, torch.Tensor) \
            else torch.tensor(meta_node_id, dtype=torch.long)

        self.value_graph.edata['meta_edge_id'] = meta_edge_id if isinstance(meta_edge_id, torch.Tensor) \
            else torch.tensor(meta_edge_id, dtype=torch.long)

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
        return 'Meta{}\n{}'.format(self.meta_graph, self.value_graph)

    @property
    def number(self):
        return self.value_graph.ndata[NUMBER]

    @property
    def value(self):
        return self.value_graph.ndata[VALUE]

    @property
    def symbol(self):
        return self.value_graph.ndata['symbol']

    @property
    def meta_node_id(self):
        return self.value_graph.ndata['meta_node_id']

    @property
    def meta_edge_id(self):
        return self.value_graph.edata['meta_edge_id']

    @property
    def meta_symbol(self):
        return self.meta_graph.ndata['meta_symbol']

    @property
    def meta_component_id(self):
        return self.meta_graph.ndata['meta_component_id']

    @property
    def meta_value(self):
        return self.meta_graph.ndata[VALUE]

    @property
    def meta_key(self):
        return self.meta_graph.ndata[KEY]

    @property
    def key(self):
        return self.meta_key[self.meta_node_id]

    @property
    def component_id(self):
        return self.meta_component_id[self.meta_node_id]

    @property
    def meta_edge_key(self):
        return self.meta_graph.edata[KEY]

    @property
    def edge_key(self):
        return self.meta_edge_key[self.meta_edge_id]

    def target_mask(self, target_symbol: list):
        return (self.symbol.view(-1, 1) == torch.tensor(target_symbol, dtype=torch.long)).any(-1)

    def save(self, filename):
        dgl.save_graphs(
            filename, [self.meta_graph, self.value_graph],
            labels={
                'meta_symbol': self.meta_symbol,
                'meta_component_id': self.meta_component_id,
                'number': self.number,
                'symbol': self.symbol,
                'meta_node_id': self.meta_node_id,
                'meta_edge_id': self.meta_edge_id
            }
        )

    @classmethod
    def load(cls, filename):
        graphs, labels = dgl.load_graphs(filename)
        return cls(
            meta_graph=graphs[0],
            graph=graphs[1],
            meta_key=graphs[0].ndata[KEY],
            meta_value=graphs[0].ndata[VALUE],
            meta_symbol=labels['meta_symbol'],
            meta_component_id=labels['meta_component_id'],
            meta_edge_key=graphs[0].edata[KEY],
            value=graphs[1].ndata[VALUE],
            number=labels['number'],
            symbol=labels['symbol'],
            meta_node_id=labels['meta_node_id'],
            meta_edge_id=labels['meta_edge_id']
        )

    def to(self, *args, **kwargs):
        self.meta_graph.to(*args, **kwargs)
        self.value_graph.to(*args, **kwargs)
