#ifndef FCN_OF_INTEREST_HPP
#define FCN_OF_INTEREST_HPP
#include <adolc/adolc.h>
#include "Functions_ANN.hpp"

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
#define N_PARAM (					\
		 (0					\
		  + AllLayer_Init<N_H1>::nparam()	\
		  + All2all<N_H1, N_INPUT>::nparam()	\
		  + AllLayer_G<N_H1, GFCN>::nparam()	\
		  + AllLayer_Init<N_H2>::nparam()	\
		  + All2all<N_H2, N_H1>::nparam()	\
		  + AllLayer_Init<N_H3>::nparam()	   \
		  + All2all<N_H3, N_H2>::nparam()	   \
		  + AllLayer_G<N_H3, GFCN>::nparam()		\
		  + (N_INPUT*AllLayer2neuron<N_H3>::nparam())	\
		  )						\
							)
#define FLOP_ESTIMATE (						\
		       (0					\
			+ AllLayer_Init<N_H1>::nflop()		\
			+ All2all<N_H1, N_INPUT>::nflop()	\
			+ AllLayer_G<N_H1, GFCN>::nflop()	\
			+ AllLayer_Init<N_H2>::nflop()		\
			+ All2all<N_H2, N_H1>::nflop()		\
			+ AllLayer_Init<N_H3>::nflop()		\
			+ All2all<N_H3, N_H2>::nflop()		      \
			+ AllLayer_G<N_H3, GFCN>::nflop()		\
			+ 1 + 3*N_INPUT*AllLayer2neuron<N_H3>::nflop()	\
			)						\
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
    
    T h1[N_H1];
    AllLayer_Init<N_H1>::fcn(h1, p, index);
    index += AllLayer_Init<N_H1>::nparam();
    
    All2all<N_H1,N_INPUT>::fcn(h1, I, p,index);
    index += All2all<N_H1, N_INPUT>::nparam();

    AllLayer_G<N_H1, GFCN>::fcn(h1, p, index);
    
    T h2[N_H2];
    AllLayer_Init<N_H2>::fcn(h2, p, index);
    index += AllLayer_Init<N_H2>::nparam();
    
    All2all<N_H2,N_H1>::fcn(h2, h1, p,index);
    index += All2all<N_H2, N_H1>::nparam();
    
    T h3[N_H3];
    AllLayer_Init<N_H3>::fcn(h3, p, index);
    index += AllLayer_Init<N_H3>::nparam();
    
    All2all<N_H3,N_H2>::fcn(h3, h2, p,index);
    index += All2all<N_H3, N_H2>::nparam();
    
    AllLayer_G<N_H3, GFCN>::fcn(h3, p, index);
    
    register T sum = 0.f;
    for(int to=0; to < N_INPUT; to++) {
      register T o = AllLayer2neuron<N_H3>::fcn(h3, p, index);
      index += AllLayer2neuron<N_H3>::nparam();
      
      if(IS_PRED == true) { pred[to] = o;
      } else { o -= I[to]; sum += o*o; }
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

