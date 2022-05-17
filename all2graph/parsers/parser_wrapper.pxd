cimport cython
from .graph_parser cimport GraphParser
from ..graph.raw_graph cimport RawGraph
from ..meta_struct cimport MetaStruct


cdef class ParserWrapper(MetaStruct):
    cdef:
        public dict _data_parser
        public dict _graph_parser

    @cython.locals(parser=GraphParser)
    cpdef call_graph_parser(self, RawGraph raw_graph, key=*)