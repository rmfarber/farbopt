CXX=icc
CXXFLAGS= -O3 -qopenmp  -std=c++11
CXXFLAGS += -xMIC-AVX512
#CXXFLAGS += -xhost
CXXFLAGS += -mmic
CXXFLAGS += -qopt-report=5
CXXFLAGS += -qopt-report-embed
CXXFLAGS += -fargument-noalias
#CXXFLAGS += -qopt-prefetch=5
#CXXFLAGS += -qopt-prefetch-distance=64,8 -qopt-streaming-stores always
CXXFLAGS += -D LAYOUT_SOA
CXXFLAGS += -D NDEBUG
CXXFLAGS += -fp-model fast=2 -no-prec-div

LDFLAGS = -qopenmp

INCLUDES= -I ../../lib/nlopt_icc/include -I . -I ../src -I ../../src 
LD_LIBRARIES= -L ../../lib/nlopt_icc/lib 

MPI_INCLUDES += -I/opt/intel/compilers_and_libraries_2016.4.258/linux/mpi/intel64/include
MPI_LD_LIBRARIES += -L /opt/intel/compilers_and_libraries_2017.2.174/linux/mpi/lib64 -lmpi
