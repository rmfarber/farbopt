#ifndef XOR_HPP
#define XOR_HPP
#include "Matrix.hpp"
#include "Gfcn.h"

#ifndef FCN_ATTRIBUTES
#define FCN_ATTRIBUTES ""
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
#define FLOP_ESTIMATE ( 2*(N_PARAM - N_OUTPUT-N_H1) + N_OUTPUT + 2 + N_H1*G_ESTIMATE )

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
  inline const char* name() {return "XOR function";}
  FCN_ATTRIBUTES
  inline const char* gFcnName() {return G_DESC_STRING; }
  
  template<bool IS_PRED>
  FCN_ATTRIBUTES
  inline float generic_fcn(const uint32_t exampleNumber, const REAL_T *p,
			   const Matrix<REAL_T> *I, Matrix<REAL_T> *pred)
  {
    register float h1;
    register float o;
    float in[2];
    
    in[0] = (*I)(exampleNumber,0);
    in[1] = (*I)(exampleNumber,1);
    
    h1 = p[0];
    o = p[1];
    h1 += in[0] * p[2];
    h1 += in[1] * p[3];
    h1 = G(h1);
    o += in[0] * p[4];
    o += in[1] * p[5];
    o += h1 * p[6];
    if(IS_PRED == true) {
      (*pred)(exampleNumber,0) = o;
      return 0.;
    } else  
      return (o - (*pred)(exampleNumber,0)) * (o - (*pred)(exampleNumber,0));
  }
  
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
