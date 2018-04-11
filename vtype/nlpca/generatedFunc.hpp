#ifndef PCA_HPP
#define PCA_HPP
#include "Matrix.hpp"
#include "Gfcn.h"

#ifndef FCN_ATTRIBUTES
#define FCN_ATTRIBUTES ""
#endif

#define N_INPUT (2)
#define N_H1 (10)
#define N_H2 (1)
#define N_H3 (10)
#define N_OUTPUT (0)
#define EXAMPLE_SIZE (2)
#define N_PARAM (83)
#define FLOP_ESTIMATE (128 + 20 * G_ESTIMATE)
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
   inline const char* name() {return "autoencoder 2x10x1x10x2 function";}

  FCN_ATTRIBUTES
  inline const char* gFcnName() {return G_DESC_STRING; }
  
  template<bool IS_PRED>
  FCN_ATTRIBUTES
  inline float generic_fcn(const uint32_t exampleNumber, const REAL_T *p,
                           const Matrix<REAL_T> *I, Matrix<REAL_T> *pred)

{

#if (__CUDA_ARCH__ < 600)
      return 0.;
#else

   REAL_T in[2];
   in[0] = (*I)(exampleNumber,0);
   in[1] = (*I)(exampleNumber,1);
   register REAL_T h1_0 = p[0];
   register REAL_T h1_1 = p[1];
   register REAL_T h1_2 = p[2];
   register REAL_T h1_3 = p[3];
   register REAL_T h1_4 = p[4];
   register REAL_T h1_5 = p[5];
   register REAL_T h1_6 = p[6];
   register REAL_T h1_7 = p[7];
   register REAL_T h1_8 = p[8];
   register REAL_T h1_9 = p[9];
   h1_0 = DATA_TYPE::fma(h1_0,in[0],p[10]);
   h1_1 = DATA_TYPE::fma(h1_1,in[0],p[11]);
   h1_2 = DATA_TYPE::fma(h1_2,in[0],p[12]);
   h1_3 = DATA_TYPE::fma(h1_3,in[0],p[13]);
   h1_4 = DATA_TYPE::fma(h1_4,in[0],p[14]);
   h1_5 = DATA_TYPE::fma(h1_5,in[0],p[15]);
   h1_6 = DATA_TYPE::fma(h1_6,in[0],p[16]);
   h1_7 = DATA_TYPE::fma(h1_7,in[0],p[17]);
   h1_8 = DATA_TYPE::fma(h1_8,in[0],p[18]);
   h1_9 = DATA_TYPE::fma(h1_9,in[0],p[19]);
   h1_0 = DATA_TYPE::fma(h1_0,in[1],p[20]);
   h1_1 = DATA_TYPE::fma(h1_1,in[1],p[21]);
   h1_2 = DATA_TYPE::fma(h1_2,in[1],p[22]);
   h1_3 = DATA_TYPE::fma(h1_3,in[1],p[23]);
   h1_4 = DATA_TYPE::fma(h1_4,in[1],p[24]);
   h1_5 = DATA_TYPE::fma(h1_5,in[1],p[25]);
   h1_6 = DATA_TYPE::fma(h1_6,in[1],p[26]);
   h1_7 = DATA_TYPE::fma(h1_7,in[1],p[27]);
   h1_8 = DATA_TYPE::fma(h1_8,in[1],p[28]);
   h1_9 = DATA_TYPE::fma(h1_9,in[1],p[29]);
   h1_0 = DATA_TYPE::vecG(h1_0);
   h1_1 = DATA_TYPE::vecG(h1_1);
   h1_2 = DATA_TYPE::vecG(h1_2);
   h1_3 = DATA_TYPE::vecG(h1_3);
   h1_4 = DATA_TYPE::vecG(h1_4);
   h1_5 = DATA_TYPE::vecG(h1_5);
   h1_6 = DATA_TYPE::vecG(h1_6);
   h1_7 = DATA_TYPE::vecG(h1_7);
   h1_8 = DATA_TYPE::vecG(h1_8);
   h1_9 = DATA_TYPE::vecG(h1_9);
   register REAL_T h2_0 = p[30];
   h2_0 = DATA_TYPE::fma(h2_0,h1_0,p[31]);
   h2_0 = DATA_TYPE::fma(h2_0,h1_1,p[32]);
   h2_0 = DATA_TYPE::fma(h2_0,h1_2,p[33]);
   h2_0 = DATA_TYPE::fma(h2_0,h1_3,p[34]);
   h2_0 = DATA_TYPE::fma(h2_0,h1_4,p[35]);
   h2_0 = DATA_TYPE::fma(h2_0,h1_5,p[36]);
   h2_0 = DATA_TYPE::fma(h2_0,h1_6,p[37]);
   h2_0 = DATA_TYPE::fma(h2_0,h1_7,p[38]);
   h2_0 = DATA_TYPE::fma(h2_0,h1_8,p[39]);
   h2_0 = DATA_TYPE::fma(h2_0,h1_9,p[40]);
   register REAL_T h3_0 = p[41];
   register REAL_T h3_1 = p[42];
   register REAL_T h3_2 = p[43];
   register REAL_T h3_3 = p[44];
   register REAL_T h3_4 = p[45];
   register REAL_T h3_5 = p[46];
   register REAL_T h3_6 = p[47];
   register REAL_T h3_7 = p[48];
   register REAL_T h3_8 = p[49];
   register REAL_T h3_9 = p[50];
   h3_0 = DATA_TYPE::fma(h3_0,h2_0,p[51]);
   h3_1 = DATA_TYPE::fma(h3_1,h2_0,p[52]);
   h3_2 = DATA_TYPE::fma(h3_2,h2_0,p[53]);
   h3_3 = DATA_TYPE::fma(h3_3,h2_0,p[54]);
   h3_4 = DATA_TYPE::fma(h3_4,h2_0,p[55]);
   h3_5 = DATA_TYPE::fma(h3_5,h2_0,p[56]);
   h3_6 = DATA_TYPE::fma(h3_6,h2_0,p[57]);
   h3_7 = DATA_TYPE::fma(h3_7,h2_0,p[58]);
   h3_8 = DATA_TYPE::fma(h3_8,h2_0,p[59]);
   h3_9 = DATA_TYPE::fma(h3_9,h2_0,p[60]);
   h3_0 = DATA_TYPE::vecG(h3_0);
   h3_1 = DATA_TYPE::vecG(h3_1);
   h3_2 = DATA_TYPE::vecG(h3_2);
   h3_3 = DATA_TYPE::vecG(h3_3);
   h3_4 = DATA_TYPE::vecG(h3_4);
   h3_5 = DATA_TYPE::vecG(h3_5);
   h3_6 = DATA_TYPE::vecG(h3_6);
   h3_7 = DATA_TYPE::vecG(h3_7);
   h3_8 = DATA_TYPE::vecG(h3_8);
   h3_9 = DATA_TYPE::vecG(h3_9);
   register REAL_T o;
   float sum = 0.f;
   o = p[61];
    o = DATA_TYPE::fma(o,h3_0,p[62]);
    o = DATA_TYPE::fma(o,h3_1,p[63]);
    o = DATA_TYPE::fma(o,h3_2,p[64]);
    o = DATA_TYPE::fma(o,h3_3,p[65]);
    o = DATA_TYPE::fma(o,h3_4,p[66]);
    o = DATA_TYPE::fma(o,h3_5,p[67]);
    o = DATA_TYPE::fma(o,h3_6,p[68]);
    o = DATA_TYPE::fma(o,h3_7,p[69]);
    o = DATA_TYPE::fma(o,h3_8,p[70]);
    o = DATA_TYPE::fma(o,h3_9,p[71]);
   if(IS_PRED == true) {
      (*pred)(exampleNumber,0) = o;
   }
   o = DATA_TYPE::sub(o, in[0]);
   sum += DATA_TYPE::reduce(DATA_TYPE::mult(o,o));
   o = p[72];
    o = DATA_TYPE::fma(o,h3_0,p[73]);
    o = DATA_TYPE::fma(o,h3_1,p[74]);
    o = DATA_TYPE::fma(o,h3_2,p[75]);
    o = DATA_TYPE::fma(o,h3_3,p[76]);
    o = DATA_TYPE::fma(o,h3_4,p[77]);
    o = DATA_TYPE::fma(o,h3_5,p[78]);
    o = DATA_TYPE::fma(o,h3_6,p[79]);
    o = DATA_TYPE::fma(o,h3_7,p[80]);
    o = DATA_TYPE::fma(o,h3_8,p[81]);
    o = DATA_TYPE::fma(o,h3_9,p[82]);
   if(IS_PRED == true) {
      (*pred)(exampleNumber,1) = o;
      return 0.;
   }
   o = DATA_TYPE::sub(o, in[1]);
   sum += DATA_TYPE::reduce(DATA_TYPE::mult(o,o));
   return(sum);
#endif
}


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


