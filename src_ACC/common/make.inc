CXX=pgc++
CXXFLAGS= -O3 
#CXXFLAGS= -g
CXXFLAGS += -acc
CXXFLAGS += -std=c++11 -DLAYOUT_SOA
CXXFLAGS += -Minfo=accel -use-fast-math
CXXFLAGS += -ta=multicore,tesla:cc35,nordc
CXXFLAGS += -D NDEBUG

CFLAGS = -std=c99
INCLUDES= -I ../../lib/nlopt_gcc/include -I . -I ../src -I ../../src
LD_LIBRARIES= -L ../../lib/nlopt_gcc/lib -L /opt/pgi/linux86-64/2016/lib/