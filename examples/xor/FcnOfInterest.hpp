#ifndef XOR_HPP
#define XOR_HPP

#ifdef USE_GRAD
#include <adolc/adolc.h>
#endif

#include "FcnOfInterest_config.h"
#include "Functions_ANN.hpp"


#ifndef FCN_ATTRIBUTES
#define FCN_ATTRIBUTES
#endif


#define N_INPUT 2
#define N_OUTPUT 1
#define N_H1 1

// Number of Parameters
// (N_OUTPUT + N_H1) // Neuron Offsets
//  + (N_INPUT*N_H1) // connections from I to H1
//  + (N_H1*N_OUTPUT) // connections from H1 to O
//  + (N_INPUT*N_OUTPUT); // connections from I to O
#define N_PARAM  ( (N_OUTPUT + N_H1) + (N_INPUT*N_H1) + (N_H1*N_OUTPUT )+ (N_INPUT*N_OUTPUT) ) 

//The final 2 ops are for the sum in the objective function
#define FLOP_ESTIMATE ( 2*(N_PARAM - N_OUTPUT-N_H1) + N_OUTPUT + 2 + N_H1*GFCN::nflop() )

template<typename REAL_T>
struct FcnOfInterest {
  FCN_ATTRIBUTES
  inline uint32_t nInput() {return N_INPUT;}
  FCN_ATTRIBUTES
  inline uint32_t nOutput() {return N_OUTPUT;}
  FCN_ATTRIBUTES
  inline uint32_t nParam() { return N_PARAM; }
  FCN_ATTRIBUTES
  inline uint32_t nFlop() {return FLOP_ESTIMATE;}
  FCN_ATTRIBUTES
  inline const char* name() {return "XOR function";}
  FCN_ATTRIBUTES
  inline const char* gFcnName() {return GFCN::name(); }
  
  template<bool IS_PRED, typename T=REAL_T>
  FCN_ATTRIBUTES
  inline T generic_fcn(const T *p, const T *I, T *pred)
  {
    register T h1;
    register T o;
    
    h1 = p[0];
    o = p[1];
    h1 += I[0] * p[2];
    h1 += I[1] * p[3];
    h1 = GFCN::fcn(h1);
    o += I[0] * p[4];
    o += I[1] * p[5];
    o += h1 * p[6];
    if(IS_PRED == true) {
      pred[0] = o;
      return 0.;
    } else  
      return (o - pred[0]) * (o - pred[0]);
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

#ifdef USE_GRAD
  FCN_ATTRIBUTES
  inline adouble CalcErr(const adouble *p, const adouble *I, const adouble *Known)
  {
    return generic_fcn<false,adouble>(p, I, const_cast< adouble *>(Known));
  }
#endif
};
#endif
