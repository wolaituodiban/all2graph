from typing import Set

from ..graph import Graph
from ..meta_struct import MetaStruct


class PostParser(MetaStruct):
    def __init__(self, add_self_loop=True, to_simple=False, to_bidirected=False,
                 seq_keys: Set[int] = None, degree=0, r_degree=0,):
        super().__init__(initialized=True)
        self.add_self_loop = add_self_loop
        self.to_bidirectied = to_bidirected
        self.seq_keys = seq_keys
        self.degree = degree
        self.r_degree = r_degree
        self.to_simple = to_simple

    def __call__(self, graph: Graph) -> Graph:
        if self.add_self_loop:
            graph = graph.add_self_loop()
        if self.to_bidirectied:
            graph = graph.to_bidirectied()
        if self.degree != 0 and self.r_degree != 0:
            graph = graph.add_edges_by_key(keys=self.seq_keys, degree=self.degree, r_degree=self.r_degree)
        if self.to_simple:
            graph = graph.to_simple()
        return graph

    def extra_repr(self) -> str:
        return 'add_self_loop={}, to_simple={}, to_bidirected={}, degree={}, r_degree={}'.format(
            self.add_self_loop, self.to_simple, self.to_bidirectied, self.degree, self.r_degree
        )
