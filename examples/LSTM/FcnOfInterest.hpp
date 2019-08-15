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

// H_LEN is the number of neurons in the hidden layer
template<typename T, int H_LEN, int X_LEN=1> class LSTM {
private:
  uint32_t nparam;
  
  FCN_ATTRIBUTES
  LSTM() { } // no void constructor
  T Wxi[H_LEN*X_LEN], Whi[H_LEN], Wci[H_LEN], Bi[H_LEN];
  T Wxf[H_LEN*X_LEN], Whf[H_LEN], Bf[H_LEN];
  T Wcf[H_LEN], Wxc[H_LEN*X_LEN], Whc[H_LEN], Bc[H_LEN]; 
  T Wxo[H_LEN*X_LEN], Who[H_LEN], Wco[H_LEN], Bo[H_LEN];
  
public:
  T input[H_LEN];
  T forget[H_LEN];
  T cell[H_LEN];
  T output[H_LEN];
  T hidden[H_LEN];
  
  FCN_ATTRIBUTES
  inline static const char *name()  {
    return("LSTM\nCitation: https://arxiv.org/pdf/1303.5778.pdf\n");
  }
  FCN_ATTRIBUTES
  inline int nParam() { return nparam;} 
  FCN_ATTRIBUTES
  inline int nFlop() { return(0); }

  FCN_ATTRIBUTES
  LSTM(int index, const T *p) {
    int oldIndex=index;
    //t==0 do initializations
    for(int i=0; i< H_LEN; ++i) {
      Whi[i] = p[index++]; Wci[i] = p[index++]; Bi[i] = p[index++];
      Whf[i] = p[index++]; Bf[i] = p[index++];
      Wcf[i] = p[index++]; Whc[i] = p[index++];
      Who[i] = p[index++]; Wco[i] = p[index++]; Bo[i] = p[index++];
      // cell and hidden defined to be the biases at time 0
      cell[i] = Bc[i] = p[index++];
      hidden[i] = p[index++];
      for(int j=0; j < X_LEN; j++) {
	Wxi[i*X_LEN+j] = p[index++];
	Wxf[i*X_LEN+j] = p[index++];
	Wxc[i*X_LEN+j] = p[index++];
	Wxo[i*X_LEN+j] = p[index++];
      }
    }
    nparam = index - oldIndex;
  }
  
  FCN_ATTRIBUTES
  inline void fcn(const T *x) {
    for(int i=0; i < H_LEN; ++i) {
      T tmp;
      // eqn (3)
      tmp=0.; for(int j=0; j < X_LEN; ++j) tmp += Wxi[i*X_LEN + j] *x[j];
      input[i] = Logistic_G::fcn(tmp + Whi[i] * hidden[i] + Wci[i]*cell[i] + Bi[i]);
      // eqn (4)
      tmp=0.; for(int j=0; j < X_LEN; ++j) tmp += Wxf[i*X_LEN + j] *x[j];
      forget[i] = Logistic_G::fcn(tmp + Whf[i] * hidden[i] + Wcf[i]*cell[i] + Bf[i]);
      // eqn (5)
      tmp=0.; for(int j=0; j < X_LEN; ++j) tmp += Wxc[i*X_LEN + j] *x[j];
      cell[i] = forget[i]*cell[i] + input[i] * Tanh_G::fcn(tmp + Bc[i]);
      // eqn (6)
      tmp=0.; for(int j=0; j < X_LEN; ++j) tmp += Wxo[i*X_LEN + j];
      output[i] = Logistic_G::fcn(tmp + Who[i] * hidden[i] + Wco[i]*cell[i] + Bo[i]);
      // eqn (7)
      hidden[i] = output[i] * Tanh_G::fcn(cell[i]);
    } 
  } 
};


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
    sprintf(name,"LSTM test");
    return name;
  }
  
  FCN_ATTRIBUTES
  inline const char* gFcnName() {return "LSTM gfunctions"; }
  
  template<bool IS_PRED, typename T=REAL_T>
  FCN_ATTRIBUTES
  inline T generic_fcn(const T *p, const T *I, T *pred)
  {
    int index=0;

    LSTM<T,LSTM_H_LEN,LSTM_X_LEN> lstm(index, p);
    index += lstm.nParam();

    // define lstm.hidden to output
    T Who[LSTM_H_LEN];
    for(int i=0; i < LSTM_H_LEN; i++) { Who[i] = p[index++]; }

    T Bo=p[index++];

    T err=0.;
    // present
    for(int i=0; i < N_INPUT; i += LSTM_X_LEN) lstm.fcn(&I[i]);

    // recall
    T bogus[N_INPUT]; for(int i=0; i < N_INPUT; ++i) bogus[i]=0.;
    for(int i=0; i < N_INPUT; i += LSTM_X_LEN) {
      lstm.fcn( ((const T *) &bogus) );
      
      // calculate output from lstm.hidden
      T o=Bo;
      for(int j=0; j < LSTM_H_LEN; j++) { o += Who[j] * lstm.hidden[j]; }
      o = Linear_G::fcn(o);
      //Softmax<T,5>::fcn(tmp);
      if(IS_PRED == true) {
	pred[i] = o;
      } else {
	T d = o - pred[i];
	err += d*d;
      }
    }
    // for bootstrap
    if(nparam == 0) nparam=index;
    return(err);
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

