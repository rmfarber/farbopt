#ifndef PCA_HPP
#define PCA_HPP
#include "Matrix.hpp"
#ifdef USE_GRAD
#include <adolc/adolc.h>
#endif
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
  inline float generic_fcn(const REAL_T *p, const REAL_T *I, REAL_T *pred)

{
   float in[2];
   in[0] = I[0];
   in[1] = I[1];
   register float h1_0 = p[0];
   register float h1_1 = p[1];
   register float h1_2 = p[2];
   register float h1_3 = p[3];
   register float h1_4 = p[4];
   register float h1_5 = p[5];
   register float h1_6 = p[6];
   register float h1_7 = p[7];
   register float h1_8 = p[8];
   register float h1_9 = p[9];
   h1_0 += in[0] * p[10];
   h1_1 += in[0] * p[11];
   h1_2 += in[0] * p[12];
   h1_3 += in[0] * p[13];
   h1_4 += in[0] * p[14];
   h1_5 += in[0] * p[15];
   h1_6 += in[0] * p[16];
   h1_7 += in[0] * p[17];
   h1_8 += in[0] * p[18];
   h1_9 += in[0] * p[19];
   h1_0 += in[1] * p[20];
   h1_1 += in[1] * p[21];
   h1_2 += in[1] * p[22];
   h1_3 += in[1] * p[23];
   h1_4 += in[1] * p[24];
   h1_5 += in[1] * p[25];
   h1_6 += in[1] * p[26];
   h1_7 += in[1] * p[27];
   h1_8 += in[1] * p[28];
   h1_9 += in[1] * p[29];
   h1_0 = G(h1_0);
   h1_1 = G(h1_1);
   h1_2 = G(h1_2);
   h1_3 = G(h1_3);
   h1_4 = G(h1_4);
   h1_5 = G(h1_5);
   h1_6 = G(h1_6);
   h1_7 = G(h1_7);
   h1_8 = G(h1_8);
   h1_9 = G(h1_9);
   register float h2_0 = p[30];
   h2_0 += h1_0 * p[31];
   h2_0 += h1_1 * p[32];
   h2_0 += h1_2 * p[33];
   h2_0 += h1_3 * p[34];
   h2_0 += h1_4 * p[35];
   h2_0 += h1_5 * p[36];
   h2_0 += h1_6 * p[37];
   h2_0 += h1_7 * p[38];
   h2_0 += h1_8 * p[39];
   h2_0 += h1_9 * p[40];
   register float h3_0 = p[41];
   register float h3_1 = p[42];
   register float h3_2 = p[43];
   register float h3_3 = p[44];
   register float h3_4 = p[45];
   register float h3_5 = p[46];
   register float h3_6 = p[47];
   register float h3_7 = p[48];
   register float h3_8 = p[49];
   register float h3_9 = p[50];
   h3_0 += h2_0 * p[51];
   h3_1 += h2_0 * p[52];
   h3_2 += h2_0 * p[53];
   h3_3 += h2_0 * p[54];
   h3_4 += h2_0 * p[55];
   h3_5 += h2_0 * p[56];
   h3_6 += h2_0 * p[57];
   h3_7 += h2_0 * p[58];
   h3_8 += h2_0 * p[59];
   h3_9 += h2_0 * p[60];
   h3_0 = G(h3_0);
   h3_1 = G(h3_1);
   h3_2 = G(h3_2);
   h3_3 = G(h3_3);
   h3_4 = G(h3_4);
   h3_5 = G(h3_5);
   h3_6 = G(h3_6);
   h3_7 = G(h3_7);
   h3_8 = G(h3_8);
   h3_9 = G(h3_9);
   register float o,sum = 0.f;
   o = p[61];
   o += h3_0 * p[62];
   o += h3_1 * p[63];
   o += h3_2 * p[64];
   o += h3_3 * p[65];
   o += h3_4 * p[66];
   o += h3_5 * p[67];
   o += h3_6 * p[68];
   o += h3_7 * p[69];
   o += h3_8 * p[70];
   o += h3_9 * p[71];
   if(IS_PRED == true) {
      pred[0] = o;
   }
   o -= in[0];
   sum += o*o;
   o = p[72];
   o += h3_0 * p[73];
   o += h3_1 * p[74];
   o += h3_2 * p[75];
   o += h3_3 * p[76];
   o += h3_4 * p[77];
   o += h3_5 * p[78];
   o += h3_6 * p[79];
   o += h3_7 * p[80];
   o += h3_8 * p[81];
   o += h3_9 * p[82];
   if(IS_PRED == true) {
      pred[1] = o;
      return 0.;
   }
   o -= in[1];
   sum += o*o;
   return(sum);
}


  adouble ad_fcn(const uint32_t exampleNumber, const adouble *p,
                           const Matrix<REAL_T> *I, Matrix<REAL_T> *pred)

{
   adouble in[2];
   in[0] = mkparam( (*I)(exampleNumber,0) );
   in[1] = mkparam( (*I)(exampleNumber,1) );
   adouble h1_0 = p[0];
   adouble h1_1 = p[1];
   adouble h1_2 = p[2];
   adouble h1_3 = p[3];
   adouble h1_4 = p[4];
   adouble h1_5 = p[5];
   adouble h1_6 = p[6];
   adouble h1_7 = p[7];
   adouble h1_8 = p[8];
   adouble h1_9 = p[9];
   h1_0 += in[0] * p[10];
   h1_1 += in[0] * p[11];
   h1_2 += in[0] * p[12];
   h1_3 += in[0] * p[13];
   h1_4 += in[0] * p[14];
   h1_5 += in[0] * p[15];
   h1_6 += in[0] * p[16];
   h1_7 += in[0] * p[17];
   h1_8 += in[0] * p[18];
   h1_9 += in[0] * p[19];
   h1_0 += in[1] * p[20];
   h1_1 += in[1] * p[21];
   h1_2 += in[1] * p[22];
   h1_3 += in[1] * p[23];
   h1_4 += in[1] * p[24];
   h1_5 += in[1] * p[25];
   h1_6 += in[1] * p[26];
   h1_7 += in[1] * p[27];
   h1_8 += in[1] * p[28];
   h1_9 += in[1] * p[29];
   h1_0 = G_ad(h1_0);
   h1_1 = G_ad(h1_1);
   h1_2 = G_ad(h1_2);
   h1_3 = G_ad(h1_3);
   h1_4 = G_ad(h1_4);
   h1_5 = G_ad(h1_5);
   h1_6 = G_ad(h1_6);
   h1_7 = G_ad(h1_7);
   h1_8 = G_ad(h1_8);
   h1_9 = G_ad(h1_9);
   adouble h2_0 = p[30];
   h2_0 += h1_0 * p[31];
   h2_0 += h1_1 * p[32];
   h2_0 += h1_2 * p[33];
   h2_0 += h1_3 * p[34];
   h2_0 += h1_4 * p[35];
   h2_0 += h1_5 * p[36];
   h2_0 += h1_6 * p[37];
   h2_0 += h1_7 * p[38];
   h2_0 += h1_8 * p[39];
   h2_0 += h1_9 * p[40];
   adouble h3_0 = p[41];
   adouble h3_1 = p[42];
   adouble h3_2 = p[43];
   adouble h3_3 = p[44];
   adouble h3_4 = p[45];
   adouble h3_5 = p[46];
   adouble h3_6 = p[47];
   adouble h3_7 = p[48];
   adouble h3_8 = p[49];
   adouble h3_9 = p[50];
   h3_0 += h2_0 * p[51];
   h3_1 += h2_0 * p[52];
   h3_2 += h2_0 * p[53];
   h3_3 += h2_0 * p[54];
   h3_4 += h2_0 * p[55];
   h3_5 += h2_0 * p[56];
   h3_6 += h2_0 * p[57];
   h3_7 += h2_0 * p[58];
   h3_8 += h2_0 * p[59];
   h3_9 += h2_0 * p[60];
   h3_0 = G_ad(h3_0);
   h3_1 = G_ad(h3_1);
   h3_2 = G_ad(h3_2);
   h3_3 = G_ad(h3_3);
   h3_4 = G_ad(h3_4);
   h3_5 = G_ad(h3_5);
   h3_6 = G_ad(h3_6);
   h3_7 = G_ad(h3_7);
   h3_8 = G_ad(h3_8);
   h3_9 = G_ad(h3_9);
   adouble o,sum = 0.f;
   o = p[61];
   o += h3_0 * p[62];
   o += h3_1 * p[63];
   o += h3_2 * p[64];
   o += h3_3 * p[65];
   o += h3_4 * p[66];
   o += h3_5 * p[67];
   o += h3_6 * p[68];
   o += h3_7 * p[69];
   o += h3_8 * p[70];
   o += h3_9 * p[71];
   o -= in[0];
   sum += o*o;
   o = p[72];
   o += h3_0 * p[73];
   o += h3_1 * p[74];
   o += h3_2 * p[75];
   o += h3_3 * p[76];
   o += h3_4 * p[77];
   o += h3_5 * p[78];
   o += h3_6 * p[79];
   o += h3_7 * p[80];
   o += h3_8 * p[81];
   o += h3_9 * p[82];
   o -= in[1];
   sum += o*o;
   return(sum);
}


  FCN_ATTRIBUTES
  inline void CalcOutput(const float *p, const REAL_T *I, REAL_T *pred)
  {
    generic_fcn<true>(p, I, pred);
  }
  
#pragma omp declare simd
  FCN_ATTRIBUTES
  inline float CalcOpt(const float *p, const REAL_T *I, const REAL_T *Known)
  {
    return generic_fcn<false>(p, I, const_cast< REAL_T *>(Known));
  }
};
#endif


