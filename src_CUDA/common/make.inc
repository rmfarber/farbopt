CXX=nvcc
CXXFLAGS= -O3 -w -Xcompiler="-fopenmp " -std=c++11 
CXXFLAGS += -D LAYOUT_SOA
CXXFLAGS += -gencode arch=compute_52,code=sm_52
CXXFLAGS += -x cu -D FCN_ATTRIBUTES="__host__ __device__"
CXXFLAGS += -rdc=true
CXXFLAGS += -D NDEBUG
CXXFLAGS += -use_fast_math
//CXXFLAGS += -DUSE_CUDA_DEVICE=0

LDFLAGS = -Xcompiler="-fopenmp"
LDFLAGS += -arch=sm_52

INCLUDES= -I ../../lib/nlopt_nvcc/include -I . -I ../src -I ../../src -I /usr/local/cuda/include
LD_LIBRARIES= -L ../../lib/nlopt_nvcc/lib -L /usr/local/cuda/lib64 -lcuda

