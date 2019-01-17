CXX=nvcc
CXXFLAGS += -w -Xcompiler="-fopenmp " -std=c++11 
CXXFLAGS += -gencode arch=compute_52,code=sm_52
CXXFLAGS += -gencode arch=compute_53,code=sm_53
CXXFLAGS += -gencode arch=compute_60,code=sm_60
CXXFLAGS += -gencode arch=compute_61,code=sm_61
CXXFLAGS += -x cu -D FCN_ATTRIBUTES="__host__ __device__"
CXXFLAGS += -rdc=true
CXXFLAGS += -D USE_GRAD
CXXFLAGS += -D NDEBUG
CXXFLAGS += -use_fast_math
//CXXFLAGS += -DUSE_CUDA_DEVICE=0

LDFLAGS = -Xcompiler="-fopenmp" 
LDFLAGS += -arch=sm_52

INCLUDES= -I $(ARCH_DIR)/../lib/adolc_gcc/include -I $(ARCH_DIR)/../lib/nlopt_nvcc/include -I . -I $(ARCH_DIR)/src -I $(ARCH_DIR)/../src -I /usr/local/cuda/include 
LD_LIBRARIES= -L $(ARCH_DIR)/../lib/nlopt_nvcc/lib -L /usr/local/cuda/lib64 -L $(ARCH_DIR)/../lib/adolc_gcc/lib64 -lnlopt -lcuda -ladolc

# Correct and uncomment to build MPI training code
#MPI_INCLUDES += -I/opt/intel/compilers_and_libraries_2016.4.258/linux/mpi/intel64/include
#MPI_LD_LIBRARIES += -L /opt/intel/compilers_and_libraries_2017.2.174/linux/mpi/lib64 -lmpi

