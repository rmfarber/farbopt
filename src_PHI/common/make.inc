CXX=icc
CXXFLAGS= -O3 -qopenmp  -std=c++11
CXXFLAGS += -xMIC-AVX512
#CXXFLAGS += -xhost
CXXFLAGS += -mmic
CXXFLAGS += -qopt-report=5
CXXFLAGS += -D LAYOUT_SOA
CXXFLAGS += -D NDEBUG
CXXFLAGS += -fp-model fast=2

LDFLAGS = -qopenmp

INCLUDES= -I ../../lib/nlopt_icc/include -I . -I ../src -I ../../src 
LD_LIBRARIES= -L ../../lib/nlopt_icc/lib 
