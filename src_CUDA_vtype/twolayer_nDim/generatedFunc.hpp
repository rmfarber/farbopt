#ifndef PCA_HPP
#define PCA_HPP
#include "Matrix.hpp"
#include "Gfcn.h"

#ifndef FCN_ATTRIBUTES
#define FCN_ATTRIBUTES ""
#endif

#define N_INPUT (8)
#define N_H1 (10)
#define N_H2 (10)
#define N_OUTPUT (1)
#define EXAMPLE_SIZE (8)
#define N_PARAM (211)
#define FLOP_ESTIMATE (385 + 20 * G_ESTIMATE)
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
   inline const char* name() {return "twolayer 8x10x10x1 (no I->O) function";}

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
   register float h2_3 = p[93];
   register float h2_4 = p[94];
   register float h2_5 = p[95];
   register float h2_6 = p[96];
   register float h2_7 = p[97];
   register float h2_8 = p[98];
   register float h2_9 = p[99];
   h2_0 += h1_0 * p[100];
   h2_1 += h1_0 * p[101];
   h2_2 += h1_0 * p[102];
   h2_3 += h1_0 * p[103];
   h2_4 += h1_0 * p[104];
   h2_5 += h1_0 * p[105];
   h2_6 += h1_0 * p[106];
   h2_7 += h1_0 * p[107];
   h2_8 += h1_0 * p[108];
   h2_9 += h1_0 * p[109];
   h2_0 += h1_1 * p[110];
   h2_1 += h1_1 * p[111];
   h2_2 += h1_1 * p[112];
   h2_3 += h1_1 * p[113];
   h2_4 += h1_1 * p[114];
   h2_5 += h1_1 * p[115];
   h2_6 += h1_1 * p[116];
   h2_7 += h1_1 * p[117];
   h2_8 += h1_1 * p[118];
   h2_9 += h1_1 * p[119];
   h2_0 += h1_2 * p[120];
   h2_1 += h1_2 * p[121];
   h2_2 += h1_2 * p[122];
   h2_3 += h1_2 * p[123];
   h2_4 += h1_2 * p[124];
   h2_5 += h1_2 * p[125];
   h2_6 += h1_2 * p[126];
   h2_7 += h1_2 * p[127];
   h2_8 += h1_2 * p[128];
   h2_9 += h1_2 * p[129];
   h2_0 += h1_3 * p[130];
   h2_1 += h1_3 * p[131];
   h2_2 += h1_3 * p[132];
   h2_3 += h1_3 * p[133];
   h2_4 += h1_3 * p[134];
   h2_5 += h1_3 * p[135];
   h2_6 += h1_3 * p[136];
   h2_7 += h1_3 * p[137];
   h2_8 += h1_3 * p[138];
   h2_9 += h1_3 * p[139];
   h2_0 += h1_4 * p[140];
   h2_1 += h1_4 * p[141];
   h2_2 += h1_4 * p[142];
   h2_3 += h1_4 * p[143];
   h2_4 += h1_4 * p[144];
   h2_5 += h1_4 * p[145];
   h2_6 += h1_4 * p[146];
   h2_7 += h1_4 * p[147];
   h2_8 += h1_4 * p[148];
   h2_9 += h1_4 * p[149];
   h2_0 += h1_5 * p[150];
   h2_1 += h1_5 * p[151];
   h2_2 += h1_5 * p[152];
   h2_3 += h1_5 * p[153];
   h2_4 += h1_5 * p[154];
   h2_5 += h1_5 * p[155];
   h2_6 += h1_5 * p[156];
   h2_7 += h1_5 * p[157];
   h2_8 += h1_5 * p[158];
   h2_9 += h1_5 * p[159];
   h2_0 += h1_6 * p[160];
   h2_1 += h1_6 * p[161];
   h2_2 += h1_6 * p[162];
   h2_3 += h1_6 * p[163];
   h2_4 += h1_6 * p[164];
   h2_5 += h1_6 * p[165];
   h2_6 += h1_6 * p[166];
   h2_7 += h1_6 * p[167];
   h2_8 += h1_6 * p[168];
   h2_9 += h1_6 * p[169];
   h2_0 += h1_7 * p[170];
   h2_1 += h1_7 * p[171];
   h2_2 += h1_7 * p[172];
   h2_3 += h1_7 * p[173];
   h2_4 += h1_7 * p[174];
   h2_5 += h1_7 * p[175];
   h2_6 += h1_7 * p[176];
   h2_7 += h1_7 * p[177];
   h2_8 += h1_7 * p[178];
   h2_9 += h1_7 * p[179];
   h2_0 += h1_8 * p[180];
   h2_1 += h1_8 * p[181];
   h2_2 += h1_8 * p[182];
   h2_3 += h1_8 * p[183];
   h2_4 += h1_8 * p[184];
   h2_5 += h1_8 * p[185];
   h2_6 += h1_8 * p[186];
   h2_7 += h1_8 * p[187];
   h2_8 += h1_8 * p[188];
   h2_9 += h1_8 * p[189];
   h2_0 += h1_9 * p[190];
   h2_1 += h1_9 * p[191];
   h2_2 += h1_9 * p[192];
   h2_3 += h1_9 * p[193];
   h2_4 += h1_9 * p[194];
   h2_5 += h1_9 * p[195];
   h2_6 += h1_9 * p[196];
   h2_7 += h1_9 * p[197];
   h2_8 += h1_9 * p[198];
   h2_9 += h1_9 * p[199];
   h2_0 = G(h2_0);
   h2_1 = G(h2_1);
   h2_2 = G(h2_2);
   h2_3 = G(h2_3);
   h2_4 = G(h2_4);
   h2_5 = G(h2_5);
   h2_6 = G(h2_6);
   h2_7 = G(h2_7);
   h2_8 = G(h2_8);
   h2_9 = G(h2_9);
   register float o,sum = 0.f;
   o = p[200];
   o += h2_0 * p[201];
   o += h2_1 * p[202];
   o += h2_2 * p[203];
   o += h2_3 * p[204];
   o += h2_4 * p[205];
   o += h2_5 * p[206];
   o += h2_6 * p[207];
   o += h2_7 * p[208];
   o += h2_8 * p[209];
   o += h2_9 * p[210];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,0) = o;
   }
   o -= in[0];
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


