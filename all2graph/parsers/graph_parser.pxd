from ..graph.raw_graph cimport RawGraph
from ..meta_struct cimport MetaStruct


cdef class GraphParser(MetaStruct):
    cdef:
        public dict dictionary
        public dict num_ecdfs
        public object tokenizer
        public str scale_method
        public dict scale_kwargs

    cpdef call(self, RawGraph raw_graph)
