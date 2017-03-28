include ../common/make.inc
include ./make.inc

TARGETS= nloptTrain.x pred.x genData.x 
MPI_TARGETS= mpiNloptTrain.x

VPATH=../../src
OBJS = $(TARGETS:.x=.o)

all: $(TARGETS)
ifdef MPI_INCLUDES
	make mpi
endif

mpi: $(MPI_TARGETS)

%.o: %.c Makefile ../common/make.inc ./make.inc
	$(CC) -std=c99 -c  $< 
%.o: %.cc Makefile ../common/make.inc ./make.inc
	$(CXX) -c $(CXXFLAGS) $(INCLUDES)  $< 
%.x: %.o
	$(CXX) $(LDFLAGS) $(LD_LIBRARIES) $< -o $@ -lnlopt

%.d: %.cc
	@g++ -fopenmp $(INCLUDES) -MM -MP -MT $*.o $< -o $@

%.d: %.c
	@gcc -std=c99 -fopenmp -MM -MP -MT $*.o $< -o $@

-include $(OBJS:%.o=%.d)

mpiNloptTrain.o: nloptTrain.cc
	$(CXX) -D USE_MPI $(CXXFLAGS) $(INCLUDES) $(MPI_INCLUDES)  ../../src/nloptTrain.cc -c -o mpiNloptTrain.o
mpiNloptTrain.x: mpiNloptTrain.o
	$(CXX) $(LDFLAGS) $(LD_LIBRARIES) $(MPI_LD_LIBRARIES) $< -o $@ -lnlopt 

clean:
	$(RM) *.x *.dat *.d *.o 
allclean:
	$(RM) *~ *.csv
	-make clean

