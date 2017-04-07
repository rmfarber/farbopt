CXX=g++
CXXFLAGS= -O3 -fopenmp  -std=c++11
#CXXFLAGS += -fopt-info-vec-missed
CXXFLAGS += -fopt-info
CXXFLAGS += -D LAYOUT_SOA
#CXXFLAGS += -D NDEBUG

LDFLAGS = -fopenmp

INCLUDES= -I ../../lib/nlopt_gcc/include -I . -I ../src -I ../../src  
LD_LIBRARIES= -L ../../lib/nlopt_gcc/lib

INCLUDES= -I ../../lib/nlopt_gcc/include -I . -I ../src -I ../../src
LD_LIBRARIES= -L ../../lib/nlopt_gcc/lib

MPI_INCLUDES +=  -I/usr/lib/openmpi/include

MPI_LD_LIBRARIES += -L ../../lib/nlopt_gcc/lib -L /usr/lib/openmpi/lib -lmpi_cxx -lmpi

