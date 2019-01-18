CXX=g++
CXXFLAGS += -fopenmp  -std=c++11
CXXFLAGS += -fopt-info-vec
CXXFLAGS += -fopt-info
CXXFLAGS += -D USE_GRAD
CXXFLAGS += -march=native
CXXFLAGS += -finline-limit=100000000
CXXFLAGS += -ffast-math
#CXXFLAGS += -D NDEBUG

LDFLAGS = -fopenmp
LDFLAGS += -L $(ARCH_DIR)/../lib/nlopt_gcc/lib -L $(ARCH_DIR)/../lib/adolc_gcc/lib64

INCLUDES= -I $(ARCH_DIR)/../lib/adolc_gcc/include -I $(ARCH_DIR)/../lib/nlopt_gcc/include -I . -I $(ARCH_DIR)/src -I $(ARCH_DIR)/../src  
LD_LIBRARIES= -lnlopt -ladolc

#Set correct paths and uncomment to build MPI training code
#MPI_INCLUDES +=  -I/usr/lib/openmpi/include
#MPI_LD_LIBRARIES += -L ../../lib/nlopt_gcc/lib -L /usr/lib/openmpi/lib -lmpi_cxx -lmpi

