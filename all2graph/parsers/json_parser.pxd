cimport cython
from .data_parser cimport DataParser
from ..graph.raw_graph cimport RawGraph


cdef class JsonParser(DataParser):
    cdef:
        public bint dense_dict
        public int dict_degree
        public int list_degree
        public set local_foreign_key_types
        public set global_foreign_key_types
        public processor

    @cython.locals(key=str, nid=cython.Py_ssize_t)
    cpdef void _add_dict(self, RawGraph graph, Py_ssize_t sample, dict obj, list vids)

    cpdef void _add_list(self, RawGraph graph, Py_ssize_t sample, str key, list obj, list vids)

    cpdef void add_obj(self, RawGraph graph, Py_ssize_t sample, obj, str key=*, vids=*)
