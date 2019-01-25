#change ARCH_DIR to build outside of the source tree
#ARCH_DIR=$(HOME)/new/farbopt/src_CUDA

ARCH_DIR=..
#CXXFLAGS = -O3 -D GFCN=Elliott_nn
#CXXFLAGS = -O3 -D GFCN=Tanh_nn
#CXXFLAGS = -O3 -D GFCN=Logistic_nn
#CXXFLAGS = -O3 -D GFCN=Elu_nn 
CXXFLAGS = -O3 -D GFCN=Linear_nn
