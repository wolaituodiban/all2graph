from ..graph.raw_graph cimport RawGraph
from ..meta_struct cimport MetaStruct


cdef class DataParser(MetaStruct):
    cdef public str data_col
    cdef public str time_col
    cdef public str time_format
    cdef public set targets
