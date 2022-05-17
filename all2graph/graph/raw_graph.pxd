cimport cython
from ..meta_struct cimport MetaStruct


cdef class RawGraph(MetaStruct):
    cdef public list samples
    cdef public list types
    cdef public list values
    cdef public list srcs
    cdef public list dsts
    cdef public set targets
    cdef dict _local_foreign_keys
    cdef dict _global_foreign_keys

    cpdef void add_edge_(self, Py_ssize_t u, Py_ssize_t v)

    cpdef void add_edges_(self, list u, list v)

    @cython.locals(u=cython.Py_ssize_t, v=cython.Py_ssize_t)
    cpdef void add_dense_edges_(self, list x)

    cpdef Py_ssize_t add_kv_(self, Py_ssize_t sample, str key, value)

    @cython.locals(number=cython.float, )
    cpdef formatted_values(self)

    @cython.locals(i=cython.Py_ssize_t, sample_id=cython.Py_ssize_t, _type=str)
    cpdef seq_info(self)
