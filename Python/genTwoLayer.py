#!/usr/bin/python
# Rob Farber

import sys,cStringIO, argparse
sOut = cStringIO.StringIO()

parser = argparse.ArgumentParser()
parser.add_argument("-n_i", "--n_input", help="specify number of input neurons",
                    type=int, default=2)
parser.add_argument("-n_h1", "--n_h1", help="specify number of h1 neurons",
                    type=int, default=10)
parser.add_argument("-n_h2", "--n_h2", help="specify number of h2 neurons",
                    type=int, default=10)
parser.add_argument("-n_o", "--n_output",
                    help="specify number of Output neurons",
                    type=int,default=1)
args = parser.parse_args()

nInput=args.n_input
nH1=args.n_h1
nH2=args.n_h2
nOutput=args.n_output

#print start of struct
print """#ifndef PCA_HPP
#define PCA_HPP
#include "Matrix.hpp"
#include "Gfcn.h"

#ifndef FCN_ATTRIBUTES
#define FCN_ATTRIBUTES ""
#endif
"""

print "#define N_INPUT (%d)" % (nInput) 
print "#define N_H1 (%d)" % (nH1) 
print "#define N_H2 (%d)" % (nH2) 
print "#define N_OUTPUT (%d)" % (nOutput) 
print "#define EXAMPLE_SIZE (%d)" % (nInput) 

sys.stdout = sOut
index=0
flopEstimate=0;
gCalls=0;

print """template<typename REAL_T>
struct generatedFcnInterest {
  FCN_ATTRIBUTES
  inline uint32_t nInput() {return N_INPUT;}
  FCN_ATTRIBUTES
  inline uint32_t nOutput() {return N_OUTPUT;}
  FCN_ATTRIBUTES
  inline uint32_t nParam() { return N_PARAM; }
  FCN_ATTRIBUTES
  inline uint32_t nFlop() {return FLOP_ESTIMATE;}
  FCN_ATTRIBUTES """

print '   inline const char* name() {return "twolayer %dx%dx%dx%d (no I->O) function";}' % (nInput, nH1, nH2, nOutput) 

print """
  FCN_ATTRIBUTES
  inline const char* gFcnName() {return G_DESC_STRING; }
  
  template<bool IS_PRED>
  FCN_ATTRIBUTES
  inline float generic_fcn(const uint32_t exampleNumber, const REAL_T *p,
                           const Matrix<REAL_T> *I, Matrix<REAL_T> *pred)
"""
print "{"
print "   float in[%d];" % (nInput)

for i in range(0,nInput):
    print "   in[%d] = (*I)(exampleNumber,%d);" % (i,i)

for i in range(0,nH1):
   print "   register float h1_%d = p[%d];" % (i,index)
   index += 1

#input to h1
for i in range(0,nInput):
    for j in range(0,nH1):
        print "   h1_%d += in[%d] * p[%d];" % (j,i,index)
        index += 1
        flopEstimate += 2

for j in range(0,nH1):
    print "   h1_%d = G(h1_%d);" % (j,j)
    gCalls += 1
   
for i in range(0,nH2):
   print "   register float h2_%d = p[%d];" % (i,index)
   index += 1
   
for i in range(0,nH1):
    for j in range(0,nH2):
        print "   h2_%d += h1_%d * p[%d];" % (j,i,index)
        index += 1
        flopEstimate += 2

for j in range(0,nH2):
    print "   h2_%d = G(h2_%d);" % (j,j)
    gCalls += 1
   
print "   register float o,sum = 0.f;"

for i in range(0,nOutput):
    print "   o = p[%d];" % (index)
    index += 1
    for j in range(0,nH2):
        print "   o += h2_%d * p[%d];" % (j,index)
        index += 1
        flopEstimate += 2

    print "   o = G(o);"
    gCalls += 1

    print "   if(IS_PRED == true) {"
    print "      (*pred)(exampleNumber,%d) = o;" %(i)
    if((i+1) == nInput):
       print "      return 0.;"
    print "   }"
    print "   o -= in[%d];" % (i)
    print "   sum += o*o;"
    flopEstimate += 3

print "   return(sum);"
print "}"
print 
print """
  FCN_ATTRIBUTES
  inline void CalcOutput(const uint32_t exampleNumber, const float *p,
                         const Matrix<REAL_T> *I, Matrix<REAL_T> *pred)
  {
    generic_fcn<true>(exampleNumber, p, I, pred);
  }
  
  FCN_ATTRIBUTES
  inline float CalcOpt(const uint32_t exampleNumber, const float *p, 
                       const Matrix<REAL_T> *I, const Matrix<REAL_T> *Known)
  {
    return generic_fcn<false>(exampleNumber, p, I,
                              const_cast< Matrix<REAL_T> *>(Known));
  }
};
#endif
"""
sys.stdout = sys.__stdout__

print "#define N_PARAM (%d)" % (index) 
flopEstimate += 2 # to account for the square and global sum
print "#define FLOP_ESTIMATE (%d + %d * G_ESTIMATE)" % (flopEstimate, gCalls) 
          
print sOut.getvalue()


