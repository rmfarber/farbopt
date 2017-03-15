#ifndef PCA_HPP
#define PCA_HPP
#include "Matrix.hpp"
#include "Gfcn.h"

#ifndef FCN_ATTRIBUTES
#define FCN_ATTRIBUTES ""
#endif

#define N_INPUT (8)
#define N_H1 (10)
#define N_H2 (3)
#define N_H3 (10)
#define N_OUTPUT (0)
#define EXAMPLE_SIZE (8)
#define N_PARAM (251)
#define FLOP_ESTIMATE (466 + 20 * G_ESTIMATE)
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
   inline const char* name() {return "autoencoder 8x10x3x10x8 function";}

  FCN_ATTRIBUTES
  inline const char* gFcnName() {return G_DESC_STRING; }
  
  template<bool IS_PRED>
  FCN_ATTRIBUTES
  inline float generic_fcn(const uint32_t exampleNumber, const REAL_T *p,
                           const Matrix<REAL_T> *I, Matrix<REAL_T> *pred)

{
   float in[8];
   in[0] = (*I)(exampleNumber,0);
   in[1] = (*I)(exampleNumber,1);
   in[2] = (*I)(exampleNumber,2);
   in[3] = (*I)(exampleNumber,3);
   in[4] = (*I)(exampleNumber,4);
   in[5] = (*I)(exampleNumber,5);
   in[6] = (*I)(exampleNumber,6);
   in[7] = (*I)(exampleNumber,7);
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
   h1_0 += in[2] * p[30];
   h1_1 += in[2] * p[31];
   h1_2 += in[2] * p[32];
   h1_3 += in[2] * p[33];
   h1_4 += in[2] * p[34];
   h1_5 += in[2] * p[35];
   h1_6 += in[2] * p[36];
   h1_7 += in[2] * p[37];
   h1_8 += in[2] * p[38];
   h1_9 += in[2] * p[39];
   h1_0 += in[3] * p[40];
   h1_1 += in[3] * p[41];
   h1_2 += in[3] * p[42];
   h1_3 += in[3] * p[43];
   h1_4 += in[3] * p[44];
   h1_5 += in[3] * p[45];
   h1_6 += in[3] * p[46];
   h1_7 += in[3] * p[47];
   h1_8 += in[3] * p[48];
   h1_9 += in[3] * p[49];
   h1_0 += in[4] * p[50];
   h1_1 += in[4] * p[51];
   h1_2 += in[4] * p[52];
   h1_3 += in[4] * p[53];
   h1_4 += in[4] * p[54];
   h1_5 += in[4] * p[55];
   h1_6 += in[4] * p[56];
   h1_7 += in[4] * p[57];
   h1_8 += in[4] * p[58];
   h1_9 += in[4] * p[59];
   h1_0 += in[5] * p[60];
   h1_1 += in[5] * p[61];
   h1_2 += in[5] * p[62];
   h1_3 += in[5] * p[63];
   h1_4 += in[5] * p[64];
   h1_5 += in[5] * p[65];
   h1_6 += in[5] * p[66];
   h1_7 += in[5] * p[67];
   h1_8 += in[5] * p[68];
   h1_9 += in[5] * p[69];
   h1_0 += in[6] * p[70];
   h1_1 += in[6] * p[71];
   h1_2 += in[6] * p[72];
   h1_3 += in[6] * p[73];
   h1_4 += in[6] * p[74];
   h1_5 += in[6] * p[75];
   h1_6 += in[6] * p[76];
   h1_7 += in[6] * p[77];
   h1_8 += in[6] * p[78];
   h1_9 += in[6] * p[79];
   h1_0 += in[7] * p[80];
   h1_1 += in[7] * p[81];
   h1_2 += in[7] * p[82];
   h1_3 += in[7] * p[83];
   h1_4 += in[7] * p[84];
   h1_5 += in[7] * p[85];
   h1_6 += in[7] * p[86];
   h1_7 += in[7] * p[87];
   h1_8 += in[7] * p[88];
   h1_9 += in[7] * p[89];
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
   register float h2_0 = p[90];
   register float h2_1 = p[91];
   register float h2_2 = p[92];
   h2_0 += h1_0 * p[93];
   h2_1 += h1_0 * p[94];
   h2_2 += h1_0 * p[95];
   h2_0 += h1_1 * p[96];
   h2_1 += h1_1 * p[97];
   h2_2 += h1_1 * p[98];
   h2_0 += h1_2 * p[99];
   h2_1 += h1_2 * p[100];
   h2_2 += h1_2 * p[101];
   h2_0 += h1_3 * p[102];
   h2_1 += h1_3 * p[103];
   h2_2 += h1_3 * p[104];
   h2_0 += h1_4 * p[105];
   h2_1 += h1_4 * p[106];
   h2_2 += h1_4 * p[107];
   h2_0 += h1_5 * p[108];
   h2_1 += h1_5 * p[109];
   h2_2 += h1_5 * p[110];
   h2_0 += h1_6 * p[111];
   h2_1 += h1_6 * p[112];
   h2_2 += h1_6 * p[113];
   h2_0 += h1_7 * p[114];
   h2_1 += h1_7 * p[115];
   h2_2 += h1_7 * p[116];
   h2_0 += h1_8 * p[117];
   h2_1 += h1_8 * p[118];
   h2_2 += h1_8 * p[119];
   h2_0 += h1_9 * p[120];
   h2_1 += h1_9 * p[121];
   h2_2 += h1_9 * p[122];
   register float h3_0 = p[123];
   register float h3_1 = p[124];
   register float h3_2 = p[125];
   register float h3_3 = p[126];
   register float h3_4 = p[127];
   register float h3_5 = p[128];
   register float h3_6 = p[129];
   register float h3_7 = p[130];
   register float h3_8 = p[131];
   register float h3_9 = p[132];
   h3_0 += h2_0 * p[133];
   h3_1 += h2_0 * p[134];
   h3_2 += h2_0 * p[135];
   h3_3 += h2_0 * p[136];
   h3_4 += h2_0 * p[137];
   h3_5 += h2_0 * p[138];
   h3_6 += h2_0 * p[139];
   h3_7 += h2_0 * p[140];
   h3_8 += h2_0 * p[141];
   h3_9 += h2_0 * p[142];
   h3_0 += h2_1 * p[143];
   h3_1 += h2_1 * p[144];
   h3_2 += h2_1 * p[145];
   h3_3 += h2_1 * p[146];
   h3_4 += h2_1 * p[147];
   h3_5 += h2_1 * p[148];
   h3_6 += h2_1 * p[149];
   h3_7 += h2_1 * p[150];
   h3_8 += h2_1 * p[151];
   h3_9 += h2_1 * p[152];
   h3_0 += h2_2 * p[153];
   h3_1 += h2_2 * p[154];
   h3_2 += h2_2 * p[155];
   h3_3 += h2_2 * p[156];
   h3_4 += h2_2 * p[157];
   h3_5 += h2_2 * p[158];
   h3_6 += h2_2 * p[159];
   h3_7 += h2_2 * p[160];
   h3_8 += h2_2 * p[161];
   h3_9 += h2_2 * p[162];
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
   o = p[163];
   o += h3_0 * p[164];
   o += h3_1 * p[165];
   o += h3_2 * p[166];
   o += h3_3 * p[167];
   o += h3_4 * p[168];
   o += h3_5 * p[169];
   o += h3_6 * p[170];
   o += h3_7 * p[171];
   o += h3_8 * p[172];
   o += h3_9 * p[173];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,0) = o;
   }
   o -= in[0];
   sum += o*o;
   o = p[174];
   o += h3_0 * p[175];
   o += h3_1 * p[176];
   o += h3_2 * p[177];
   o += h3_3 * p[178];
   o += h3_4 * p[179];
   o += h3_5 * p[180];
   o += h3_6 * p[181];
   o += h3_7 * p[182];
   o += h3_8 * p[183];
   o += h3_9 * p[184];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,1) = o;
   }
   o -= in[1];
   sum += o*o;
   o = p[185];
   o += h3_0 * p[186];
   o += h3_1 * p[187];
   o += h3_2 * p[188];
   o += h3_3 * p[189];
   o += h3_4 * p[190];
   o += h3_5 * p[191];
   o += h3_6 * p[192];
   o += h3_7 * p[193];
   o += h3_8 * p[194];
   o += h3_9 * p[195];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,2) = o;
   }
   o -= in[2];
   sum += o*o;
   o = p[196];
   o += h3_0 * p[197];
   o += h3_1 * p[198];
   o += h3_2 * p[199];
   o += h3_3 * p[200];
   o += h3_4 * p[201];
   o += h3_5 * p[202];
   o += h3_6 * p[203];
   o += h3_7 * p[204];
   o += h3_8 * p[205];
   o += h3_9 * p[206];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,3) = o;
   }
   o -= in[3];
   sum += o*o;
   o = p[207];
   o += h3_0 * p[208];
   o += h3_1 * p[209];
   o += h3_2 * p[210];
   o += h3_3 * p[211];
   o += h3_4 * p[212];
   o += h3_5 * p[213];
   o += h3_6 * p[214];
   o += h3_7 * p[215];
   o += h3_8 * p[216];
   o += h3_9 * p[217];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,4) = o;
   }
   o -= in[4];
   sum += o*o;
   o = p[218];
   o += h3_0 * p[219];
   o += h3_1 * p[220];
   o += h3_2 * p[221];
   o += h3_3 * p[222];
   o += h3_4 * p[223];
   o += h3_5 * p[224];
   o += h3_6 * p[225];
   o += h3_7 * p[226];
   o += h3_8 * p[227];
   o += h3_9 * p[228];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,5) = o;
   }
   o -= in[5];
   sum += o*o;
   o = p[229];
   o += h3_0 * p[230];
   o += h3_1 * p[231];
   o += h3_2 * p[232];
   o += h3_3 * p[233];
   o += h3_4 * p[234];
   o += h3_5 * p[235];
   o += h3_6 * p[236];
   o += h3_7 * p[237];
   o += h3_8 * p[238];
   o += h3_9 * p[239];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,6) = o;
   }
   o -= in[6];
   sum += o*o;
   o = p[240];
   o += h3_0 * p[241];
   o += h3_1 * p[242];
   o += h3_2 * p[243];
   o += h3_3 * p[244];
   o += h3_4 * p[245];
   o += h3_5 * p[246];
   o += h3_6 * p[247];
   o += h3_7 * p[248];
   o += h3_8 * p[249];
   o += h3_9 * p[250];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,7) = o;
      return 0.;
   }
   o -= in[7];
   sum += o*o;
   return(sum);
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


