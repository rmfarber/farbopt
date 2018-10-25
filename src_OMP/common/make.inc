CXX=g++
CXXFLAGS += -fopenmp  -std=c++11
#CXXFLAGS += -fopt-info-vec-missed
CXXFLAGS += -fopt-info
CXXFLAGS += -D LAYOUT_SOA
#CXXFLAGS += -D NDEBUG

LDFLAGS = -fopenmp

INCLUDES= -I $(ARCH_DIR)/../lib/nlopt_gcc/include -I . -I $(ARCH_DIR)/src -I $(ARCH_DIR)/../src  
LD_LIBRARIES= -L $(ARCH_DIR)/../lib/nlopt_gcc/lib

INCLUDES= -I $(ARCH_DIR)/../lib/nlopt_gcc/include -I . -I $(ARCH_DIR)/src -I $(ARCH_DIR)/../src
LD_LIBRARIES= -L $(ARCH_DIR)/../lib/nlopt_gcc/lib

#Set correct paths and uncomment to build MPI training code
#MPI_INCLUDES +=  -I/usr/lib/openmpi/include
#MPI_LD_LIBRARIES += -L ../../lib/nlopt_gcc/lib -L /usr/lib/openmpi/lib -lmpi_cxx -lmpi

