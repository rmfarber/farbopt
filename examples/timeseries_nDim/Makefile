include ./make.inc
include ../common/make.inc

TARGETS= nloptTrain.x pred.x genMackeyGlass.x genScale.x doScale.x
MPI_TARGETS= mpiNloptTrain.x

VPATH=$(ARCH_DIR)/../src
OBJS = $(TARGETS:.x=.o)

all: $(TARGETS)
ifdef MPI_INCLUDES
	make mpi
endif

mpi: $(MPI_TARGETS)

%.o: %.c Makefile $(ARCH_DIR)/common/make.inc ./make.inc
	$(CC) -std=c99 -c  $< 
%.o: %.cc Makefile $(ARCH_DIR)/common/make.inc ./make.inc
	$(CXX) -c $(CXXFLAGS) $(INCLUDES)  $< 
%.x: %.o
	$(CXX) $(LDFLAGS) $< -o $@ $(LD_LIBRARIES)

%.d: %.cc
	@g++ -std=c++11 -fopenmp $(INCLUDES) -MM -MP -MT $*.o $< -o $@

%.d: %.c
	@gcc -std=c99 -fopenmp -MM -MP -MT $*.o $< -o $@

-include $(OBJS:%.o=%.d)

mpiNloptTrain.o: nloptTrain.cc nloptTrain.o
	$(CXX) -D USE_MPI $(CXXFLAGS) $(INCLUDES) $(MPI_INCLUDES)  $(ARCH_DIR)/../src/nloptTrain.cc -c -o mpiNloptTrain.o
mpiNloptTrain.x: mpiNloptTrain.o
	$(CXX) $(LDFLAGS) $< -o $@ $(MPI_LD_LIBRARIES) $(LD_LIBRARIES)

clean:
	$(RM) *.x *.dat *.d *.o MPI.o* *.png *.txt *.optrpt
allclean:
	$(RM) *~ *.csv
	-make clean

