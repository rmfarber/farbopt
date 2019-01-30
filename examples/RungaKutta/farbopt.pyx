# distutils: language = c++
# distutils: sources = PredFcn.cpp
# distutils: include_dirs = ../../lib/adolc_gcc/include ../../src
# distutils: depends = FcnOfInterest.hpp FcnOfInterest_config.h

# Cython interface file for wrapping the object
#
#

from libcpp.vector cimport vector

# c++ interface to cython
cdef extern from "PredFcn.h" namespace "farbopt":
  cdef cppclass PredFcn:
        loadParam(const char *) except +
        PredFcn(const char *) except +
        vector[float] predict(vector[float])

# creating a cython wrapper class
cdef class PyPredFcn:
    cdef PredFcn *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, s: bytes) -> None:
        self.thisptr = new PredFcn(s)
    def __dealloc__(self):
        del self.thisptr
    def predict(self, sv):
        return self.thisptr.predict(sv)
