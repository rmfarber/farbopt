#this is deprecated and only kept for future reference
#!/usr/bin/python
# Rob Farber

import sys,cStringIO, argparse
sOut = cStringIO.StringIO()

parser = argparse.ArgumentParser()
parser.add_argument("-n_i", "--n_input", help="specify number of input neurons",
                    type=int, default=2)
parser.add_argument("-n_h1", "--n_h1", help="specify number of h1 neurons",
                    type=int, default=10)
parser.add_argument("-n_bottle", help="specify number of bottleneck neurons",
                    type=int,default=1)
parser.add_argument("-n_h3", "--n_h3", help="specify number of h3 neurons",
                    type=int, default=10)
args = parser.parse_args()

nInput=args.n_input
nH1=args.n_h1
nH2=args.n_bottle
nH3=args.n_h3

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
print "#define N_H3 (%d)" % (nH3) 
print "#define N_OUTPUT (%d)" % (0) 
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

print '   inline const char* name() {return "autoencoder %dx%dx%dx%dx%d function";}' % (nInput, nH1, nH2, nH3, nInput) 

print """
  FCN_ATTRIBUTES
  inline const char* gFcnName() {return G_DESC_STRING; }
  
  template<bool IS_PRED>
  FCN_ATTRIBUTES
  inline float generic_fcn(const uint32_t exampleNumber, const REAL_T *p,
                           const Matrix<REAL_T> *I, Matrix<REAL_T> *pred)
"""
print "{"
print """
#if (__CUDA_ARCH__ < 600)
      return 0.;
#else
"""

print "   REAL_T in[%d];" % (nInput)

for i in range(0,nInput):
    print "   in[%d] = (*I)(exampleNumber,%d);" % (i,i)

for i in range(0,nH1):
   print "   register REAL_T h1_%d = p[%d];" % (i,index)
   index += 1

#input to h1
for i in range(0,nInput):
    for j in range(0,nH1):
        #print "   h1_%d += in[%d] * p[%d];" % (j,i,index)
        print "   h1_%d = DATA_TYPE::fma(h1_%d,in[%d],p[%d]);" % (j,j,i,index)
        index += 1
        flopEstimate += 2

for j in range(0,nH1):
    print "   h1_%d = DATA_TYPE::vecG(h1_%d);" % (j,j)
    gCalls += 1
   
for i in range(0,nH2):
   print "   register REAL_T h2_%d = p[%d];" % (i,index)
   index += 1
   
for i in range(0,nH1):
    for j in range(0,nH2):
        #print "   h2_%d += h1_%d * p[%d];" % (j,i,index)
        print "   h2_%d = DATA_TYPE::fma(h2_%d,h1_%d,p[%d]);" % (j,j,i,index)
        index += 1
        flopEstimate += 2

for i in range(0,nH3):
   print "   register REAL_T h3_%d = p[%d];" % (i,index)
   index += 1
   
for i in range(0,nH2):
    for j in range(0,nH3):
        #print "   h3_%d += h2_%d * p[%d];" % (j,i,index)
        print "   h3_%d = DATA_TYPE::fma(h3_%d,h2_%d,p[%d]);" % (j,j,i,index)
        index += 1
        flopEstimate += 2

for j in range(0,nH3):
    print "   h3_%d = DATA_TYPE::vecG(h3_%d);" % (j,j)
    gCalls += 1
   
print "   register REAL_T o;"
print "   float sum = 0.f;"

for i in range(0,nInput):
    print "   o = p[%d];" % (index)
    index += 1
    for j in range(0,nH3):
        #print "   o += h3_%d * p[%d];" % (j,index)
        print "    o = DATA_TYPE::fma(o,h3_%d,p[%d]);" % (j,index)
        index += 1
        flopEstimate += 2

    print "   if(IS_PRED == true) {"
    print "      (*pred)(exampleNumber,%d) = o;" %(i)
    if((i+1) == nInput):
       print "      return 0.;"
    print "   }"
    #print "   o -= in[%d];" % (i)
    print "   o = DATA_TYPE::sub(o, in[%d]);" % (i)
    print "   sum += DATA_TYPE::reduce(DATA_TYPE::mult(o,o));"
    flopEstimate += 3

print "   return(sum);"
print "#endif"
print "}"
print 
print """
  FCN_ATTRIBUTES
  inline void CalcOutput(const uint32_t exampleNumber, const REAL_T *p,
                         const Matrix<REAL_T> *I, Matrix<REAL_T> *pred)
  {
    generic_fcn<true>(exampleNumber, p, I, pred);
  }
  
  FCN_ATTRIBUTES
  inline float CalcOpt(const uint32_t exampleNumber, const REAL_T *p, 
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


