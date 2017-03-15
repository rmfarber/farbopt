include ../common/make.inc

TARGETS= nloptTrain.x pred.x genData.x
VPATH=../../src
OBJS = $(TARGETS:.x=.o)

all: $(TARGETS)

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

clean:
	$(RM) *.x *.dat *.d *.o 
allclean:
	$(RM) *~ *.csv
	-make clean

