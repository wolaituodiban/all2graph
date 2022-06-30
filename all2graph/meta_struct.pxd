cdef class MetaStruct:
    cdef:
        public str version
        dict __dict__
