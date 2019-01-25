#ifndef PCA_HPP
#define PCA_HPP
#include <adolc/adolc.h>
#include "Gfcn.h"

#ifndef FCN_ATTRIBUTES
#define FCN_ATTRIBUTES
#endif

// convenience to use DIM from from common.sh
// edit DEFINE_INPUTSIZE.sh to change
#include "InputSize.h"

//#define N_INPUT 16
#define N_H1 (10)
#define N_H2 (1)
#define N_H3 (10)
#define N_OUTPUT (0)
#define N_PARAM (\
  + N_H1  \
  + N_INPUT* N_H1  \
  + N_H2  \
  + N_H1* N_H2  \
  + N_H3  \
  + N_H2* N_H3  \
  + N_INPUT + N_INPUT*N_H3 \
  )
#define FLOP_ESTIMATE (				\
		       N_INPUT			\
		       + N_H1			\
		       + 2*(N_INPUT * N_H1)	\
		       + GFCN::flops()*N_H1	\
		       + N_H2			\
		       + 2*(N_H1 * N_H2)	\
		       + N_H3			\
		       + 2*(N_H2 * N_H3)	\
		       + GFCN::flops()*N_H3	\
		       + 1			\
		       + (N_INPUT * (2*N_H3 + 3) )	\
						)
template<typename REAL_T>
struct generatedFcnInterest {
  FCN_ATTRIBUTES
  inline uint32_t nInput() {return N_INPUT;}
  FCN_ATTRIBUTES
  inline uint32_t nOutput() {return N_OUTPUT;}
  FCN_ATTRIBUTES
  inline uint32_t nParam() { return N_PARAM; }
  FCN_ATTRIBUTES
  inline uint32_t nFlop() {return FLOP_ESTIMATE;}
  FCN_ATTRIBUTES 
  // really hate the following. Fix using preprocessor later.
  inline const char* name() {
    static char name[256];
    sprintf(name,"Autoencoder %dx%dx%dx%dx%d",N_INPUT,N_H1,N_H2,N_H3,N_INPUT);
    return name;
  }
  
  FCN_ATTRIBUTES
  inline const char* gFcnName() {return GFCN::name(); }
  
  template<bool IS_PRED, typename T=REAL_T>
  FCN_ATTRIBUTES
  inline T generic_fcn(const T *p, const T *I, T *pred)
    
  {
    register int index=0;
    
    // FLOP/s (+ N_H1) 
    // NPARAM (+ N_H1) 
    T h1[N_H1];
    for(int i=0; i < N_H1; i++) h1[i] = p[index++];
    
    // FLOP/s (+ 2*(N_INPUT * N_H1))
    // NPARAM (+ N_INPUT* N_H1) 
    for(int from=0; from < N_INPUT; from++) {
      for(int to=0; to < N_H1; to++) {
	h1[to] += I[from] * p[index++];
      }
    } 
    // FLOP/s (+ GFCN::flops()*N_H1)
    for(int i=0; i < N_H1; i++) h1[i] = GFCN::G(h1[i]);
    
    
    // FLOP/s (+ N_H2) 
    // NPARAM (+ N_H2) 
    T h2[N_H2];
    for(int i=0; i < N_H2; i++) h2[i] = p[index++];
    
    // FLOP/s (+ 2*(N_H1 * N_H2))
    // NPARAM (+ N_H1* N_H2) 
    for(int from=0; from < N_H1; from++) {
      for(int to=0; to < N_H2; to++) {
	h2[to] += h1[from] * p[index++];
      }
    } 
    
    T h3[N_H3];
    // FLOP/s (+ N_H3)
    // NPARAM (+ N_H3) 
    for(int i=0; i < N_H3; i++) h3[i] = p[index++];
    
    // FLOP/s (+ 2*(N_H2 * N_H3))
    // NPARAM (+ N_H2* N_H3) 
    for(int from=0; from < N_H2; from++) {
      for(int to=0; to < N_H3; to++) {
	h3[to] += h2[from] * p[index++];
      }
    } 
    
    // FLOP/s (+ GFCN::flops()*N_H3)
    for(int i=0; i < N_H3; i++) h3[i] = GFCN::G(h3[i]);
    
    // FLOP/s (+1)
    register T sum = 0.f;

    // FLOP/s (+ (N_INPUT * (2*N_H3 + 3)))
    // NPARAM (+ N_INPUT + N_INPUT*N_H3) 
    for(int to=0; to < N_INPUT; to++) {
      register T o = p[index++];
      for(int from=0; from < N_H3; from++) o += h3[from] * p[index++];
      
      if(IS_PRED == true) { pred[to] = o;
      } else {
	o -= I[to];
	sum += o*o;
      }
    }
    return(sum);
  }
  
  FCN_ATTRIBUTES
  inline void CalcOutput(const float *p, const REAL_T *I, REAL_T *pred)
  {
    generic_fcn<true,REAL_T>(p, I, pred);
  }
  
#pragma omp declare simd
  FCN_ATTRIBUTES
  inline float CalcOpt(const float *p, const REAL_T *I, const REAL_T *Known)
  {
    return generic_fcn<false,REAL_T>(p, I, const_cast< REAL_T *>(Known));
  }

  FCN_ATTRIBUTES
  inline adouble CalcErr(const adouble *p, const adouble *I, const adouble *Known)
  {
    return generic_fcn<false,adouble>(p, I, const_cast< adouble *>(Known));
  }
};
#endif

