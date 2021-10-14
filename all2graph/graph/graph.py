import dgl
import torch

from .raw import RawGraph
from ..version import __version__
from ..preserves import KEY, VALUE


class Graph:
    def __init__(self, meta_graph: RawGraph, graph: RawGraph, meta_key, meta_value, meta_symbol, meta_component_id,
                 meta_edge_key, value, number, symbol, meta_node_id, meta_edge_id):
        self.version = __version__
        self.meta_graph: dgl.DGLGraph = dgl.graph(
            data=(torch.tensor(meta_graph.src, dtype=torch.long), torch.tensor(meta_graph.dst, dtype=torch.long)),
            num_nodes=meta_graph.num_nodes
        )
        self.value_graph: dgl.DGLGraph = dgl.graph(
            data=(torch.tensor(graph.src, dtype=torch.long), torch.tensor(graph.dst, dtype=torch.long)),
            num_nodes=graph.num_nodes
        )

        assert len(meta_key) == len(meta_value) == len(meta_symbol) == len(meta_component_id) == \
               self.meta_graph.num_nodes(), (
            len(meta_key), len(meta_value), len(meta_symbol), len(meta_component_id), self.meta_graph.num_nodes()
        )
        self.meta_graph.ndata[KEY] = torch.tensor(meta_key, dtype=torch.long)
        self.meta_graph.ndata[VALUE] = torch.tensor(meta_value, dtype=torch.long)
        self.meta_symbol = torch.tensor(meta_symbol, dtype=torch.long)
        self.meta_component_id = torch.tensor(meta_component_id, dtype=torch.long)

        assert len(meta_edge_key) == self.meta_graph.num_edges()
        self.meta_graph.edata[KEY] = torch.tensor(meta_edge_key, dtype=torch.long)

        assert len(value) == len(number) == len(meta_node_id) == len(symbol) == self.value_graph.num_nodes()
        self.value_graph.ndata[VALUE] = torch.tensor(value, dtype=torch.long)
        self.number = torch.tensor(number, dtype=torch.float32)
        self.symbol = torch.tensor(symbol, dtype=torch.long)
        self.meta_node_id = torch.tensor(meta_node_id, dtype=torch.long)

        assert len(meta_edge_id) == self.value_graph.num_edges()
        self.meta_edge_id = torch.tensor(meta_edge_id, dtype=torch.long)

    def __repr__(self):
        return 'Meta{}\n{}'.format(self.meta_graph, self.value_graph)

    @property
    def meta_value(self):
        return self.meta_graph.ndata[VALUE]

    @property
    def value(self):
        return self.value_graph.ndata[VALUE]

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
