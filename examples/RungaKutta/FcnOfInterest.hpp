#ifndef FCN_OF_INTEREST_HPP
#define FCN_OF_INTEREST_HPP
#include <adolc/adolc.h>
#include "Functions_ANN.hpp"

#ifndef FCN_ATTRIBUTES
#define FCN_ATTRIBUTES
#endif

#include "FcnOfInterest_config.h"

#define N_INPUT 2
#define N_H1 (5)
#define N_H2 (5)
#define N_OUTPUT (2)

#ifndef RK4_H
#define RK4_H 0.1
#endif
#ifndef RK4_RECURRENCE_LOOPS
#define RK4_RECURRENCE_LOOPS 5
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
    for(int i=0; i < guess_nparam; i++) guess_param[i] = 0.f;
    float *I=new float[nInput()];
    for(uint32_t i=0; i < nInput(); i++) I[i] = 0.f;
    generic_fcn<false,float>(guess_param, I, I);
    delete [] guess_param;
    delete [] I;
  }

  FCN_ATTRIBUTES
  FcnOfInterest() {
    nparam=nflop=0;
    bootstrap(1000000);
    nflop = (0
	     + AllLayer_Init<N_H1>::nflop()
	     + FromAll2all<N_INPUT, N_H1>::nflop()
	     + AllLayer_G<N_H1, GFCN>::nflop()
	     + AllLayer_Init<N_H2>::nflop()
	     + FromAll2all<N_H1, N_H2>::nflop()
	     + AllLayer_G<N_H2, GFCN>::nflop()
	     + 1 + 3*N_OUTPUT*AllLayer2neuron<N_H2>::nflop()
	     );
  }

  FCN_ATTRIBUTES
  inline uint32_t nInput() {return FI_N_INPUT;}

  FCN_ATTRIBUTES
  inline uint32_t nOutput() {return FI_N_OUTPUT;}

  FCN_ATTRIBUTES
  inline uint32_t nParam() { return nparam; }

  FCN_ATTRIBUTES
  inline uint32_t nFlop() {return nflop;}

  FCN_ATTRIBUTES 
  // really hate the following. Fix using preprocessor later.
  inline const char* name() {
    static char name[10244];
#ifdef EXPLICIT_RK4
    sprintf(name,"EXPLICIT RK4 twolayer %dx%dx%dx%d\nCitation: https://arxiv.org/pdf/comp-gas/9305001.pdf\nRK4_H=%f",N_INPUT,N_H1,N_H2,N_OUTPUT,RK4_H);
#else
    sprintf(name,"IMPLICIT RK4 twolayer %dx%dx%dx%d\nCitation: https://arxiv.org/pdf/comp-gas/9305001.pdf\nRK4_H=%f",N_INPUT,N_H1,N_H2,N_OUTPUT,RK4_H);
#endif
    return name;
  }
  
  FCN_ATTRIBUTES
  inline const char* gFcnName() {return GFCN::name(); }
  
  template<bool IS_PRED, typename T=REAL_T>
  FCN_ATTRIBUTES
  inline void rhs(const T *p, const T *I, T *pred)
  {
    int index=0;
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
    
    AllLayer_G<N_H2, GFCN>::fcn(h2, p, index);
    
    for(int to=0; to < N_OUTPUT; to++) {
      T o = AllLayer2neuron<N_H2>::fcn(h2, p, index);
      index += AllLayer2neuron<N_H2>::nparam();
      pred[to] = o;
    }
    if(nparam==0) nparam = index;
  }

#ifdef EXPLICIT_RK4
  template<bool IS_PRED, typename T=REAL_T>
  FCN_ATTRIBUTES
  inline T generic_fcn(const T *p, const T *I, T *pred)
  {
    // Implement a 2D 4th Order Runga Kutta
    T h=RK4_H;
    T k1[2],k2[2],k3[2],k4[2], x[2];
    // from Python code
    //k1 = dt * f(x,t)
    //k2 = dt * f(x + x / 2.0,t)
    //k3 = dt * f(x + k2 / 2.0,t)
    //k4 = dt * f(x + k3,t)
    //x = x + ( k1 + 2.0 * k2 + 2.0 * k3 + k4 ) / 6.0

    
    rhs<true, T>(p, I, k1);
    for(int i=0; i < 2; ++i) k1[i] *= h;

    for(int i=0; i < 2; ++i) x[i] = I[i] + I[i]*0.5;
    rhs<true, T>(p, x, k2);
    for(int i=0; i < 2; ++i) k2[i] *= h;

    for(int i=0; i < 2; ++i) x[i] = I[i] + k2[i]*0.5;
    rhs<true, T>(p, x, k3);
    for(int i=0; i < 2; ++i) k3[i] *= h;

    for(int i=0; i < 2; ++i) x[i] = I[i] + k3[i];
    rhs<true, T>(p, x, k4);
    for(int i=0; i < 2; ++i) k4[i] *= h;

    for(int i=0; i < 2; ++i)
      x[i] = I[i] + ( k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i] ) / 6.0;

    T err = 0.;
    if(IS_PRED == true) {
      for(int i=0; i < 2; ++i) pred[i] = x[i];
    } else {
      for(int i=0; i < 2; ++i) {
	T d = x[i] - pred[i];
	err += d*d;
      }
    }
    return(err);
  }
#else
  template<bool IS_PRED, typename T=REAL_T>
  FCN_ATTRIBUTES
  inline T generic_fcn(const T *p, const T *I, T *pred)
  {
    T h=RK4_H;
    T t1[2],t2[2],Yn_1[2];
    
    //rhs<IS_PRED, T>(p, I, Yn_1);
    for(int i=0; i < N_INPUT; i++) Yn_1[i] = I[i];
    rhs<IS_PRED, T>(p, I, t1); // this never changes!
    for(int i=0; i < RK4_RECURRENCE_LOOPS; i++) {
      //Yn_1[1] += h; // advance to the Yn_1 state
      rhs<IS_PRED, T>(p, Yn_1, t2);
      for(int i=0; i < 2; ++i) Yn_1[i] = I[i] + h*0.5*( t1[i] + t2[i] );
    }

    T err = 0.;
    if(IS_PRED == true) {
      for(int i=0; i < 2; ++i) pred[i] = Yn_1[i];
    } else {
      for(int i=0; i < 2; ++i) {
	T d = Yn_1[i] - pred[i];
	err += d*d;
      }
    }
    return(err);
  }
#endif
  
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

  FCN_ATTRIBUTES
  inline adouble CalcErr(const adouble *p, const adouble *I, const adouble *Known)
  {
    return generic_fcn<false,adouble>(p, I, const_cast< adouble *>(Known));
  }
};
#endif

