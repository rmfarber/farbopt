#ifndef FCN_OF_INTEREST_HPP
#define FCN_OF_INTEREST_HPP


#ifdef USE_GRAD
#include <adolc/adolc.h>
#endif
#include "FcnOfInterest_config.h"
#include "Functions_ANN.hpp"

#ifndef FCN_ATTRIBUTES
#define FCN_ATTRIBUTES
#endif

template<typename REAL_T, int FI_N_INPUT=N_INPUT, int FI_N_OUTPUT=(N_OUTPUT==0)?N_INPUT:N_OUTPUT>
class FcnOfInterest {
private:
  uint32_t nparam;
  uint32_t nflop;

public:
  // do one fcn evaluation to set the nparameters
  FCN_ATTRIBUTES
  void bootstrap(int guess_nparam ) {
    float *guess_param=new float[guess_nparam];
    for(uint32_t i=0; i < guess_nparam; i++) guess_param[i] = 0.f;
    float *I=new float[nInput()];
    for(int i=0; i < nInput(); i++) I[i] = 0.f;
    generic_fcn<false,float>(guess_param, I, I);
    delete [] guess_param;
    delete [] I;
  }

  FCN_ATTRIBUTES
  FcnOfInterest() {
    nparam=nflop=0;
    bootstrap(1000000);
    nflop =
      (0
       + AllLayer_Init<N_H1>::nflop()
       + FromAll2all<N_INPUT, N_H1>::nflop()
       + AllLayer_G<N_H1, GFCN>::nflop()
       + AllLayer_Init<N_H2>::nflop()
       + FromAll2all<N_H1, N_H2>::nflop()
       + AllLayer_Init<N_H3>::nflop()
       + FromAll2all<N_H2, N_H3>::nflop()
       + AllLayer_G<N_H3, GFCN>::nflop()
       + 1 + 3*N_INPUT*AllLayer2neuron<N_H3>::nflop()
       );
  }
  FCN_ATTRIBUTES
  inline uint32_t nInput() {return FI_N_INPUT;}
  FCN_ATTRIBUTES
  inline uint32_t nOutput() {return FI_N_OUTPUT;}
  FCN_ATTRIBUTES
  inline const uint32_t nParam() { return nparam; }
  FCN_ATTRIBUTES
  inline const uint32_t nFlop() {return nflop;}
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
    
    FromAll2all<N_INPUT,N_H1>::fcn(I, h1, p,index);
    index += FromAll2all<N_INPUT, N_H1>::nparam();

    AllLayer_G<N_H1, GFCN>::fcn(h1, p, index);
    
    T h2[N_H2];
    AllLayer_Init<N_H2>::fcn(h2, p, index);
    index += AllLayer_Init<N_H2>::nparam();
    
    FromAll2all<N_H1,N_H2>::fcn(h1, h2, p,index);
    index += FromAll2all<N_H1, N_H2>::nparam();
    
    T h3[N_H3];
    AllLayer_Init<N_H3>::fcn(h3, p, index);
    index += AllLayer_Init<N_H3>::nparam();
    
    FromAll2all<N_H2,N_H3>::fcn(h2, h3, p,index);
    index += FromAll2all<N_H2, N_H3>::nparam();
    
    AllLayer_G<N_H3, GFCN>::fcn(h3, p, index);
    
    register T sum = 0.f;
    for(int to=0; to < N_INPUT; to++) {
      register T o = AllLayer2neuron<N_H3>::fcn(h3, p, index);
      index += AllLayer2neuron<N_H3>::nparam();
      
      if(IS_PRED == true) { pred[to] = o;
      } else { o -= I[to]; sum += o*o; }
    }

    // for bootstrap
    if(nparam == 0) nparam=index;
    
    return(sum);
  }
  
  FCN_ATTRIBUTES
  inline void CalcOutput(const float *p, const REAL_T *I, REAL_T *pred)
  {
    generic_fcn<true,REAL_T>(p, I, pred);
  }
  
#pragma omp declare simd
  FCN_ATTRIBUTES
  inline float CalcOpt(const float * __restrict__ p, const REAL_T * __restrict__ I, const REAL_T * __restrict__ Known)
  {
    return generic_fcn<false,REAL_T>(p, I, const_cast< REAL_T *>(Known));
  }

#ifdef USE_GRAD
  FCN_ATTRIBUTES
  inline adouble CalcErr(const adouble *p, const adouble *I, const adouble *Known)
  {
    return generic_fcn<false,adouble>(p, I, const_cast< adouble *>(Known));
  }
#endif
};
#endif

