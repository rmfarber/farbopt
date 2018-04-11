#ifndef PCA_HPP
#define PCA_HPP
#include "Matrix.hpp"
#include "Gfcn.h"

#ifndef FCN_ATTRIBUTES
#define FCN_ATTRIBUTES ""
#endif

#define N_INPUT (16)
#define N_H1 (10)
#define N_H2 (3)
#define N_H3 (10)
#define N_OUTPUT (0)
#define EXAMPLE_SIZE (16)
#define N_PARAM (419)
#define FLOP_ESTIMATE (810 + 20 * G_ESTIMATE)
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
   inline const char* name() {return "autoencoder 16x10x3x10x16 function";}

  FCN_ATTRIBUTES
  inline const char* gFcnName() {return G_DESC_STRING; }
  
  template<bool IS_PRED>
  FCN_ATTRIBUTES
  inline float generic_fcn(const uint32_t exampleNumber, const REAL_T *p,
                           const Matrix<REAL_T> *I, Matrix<REAL_T> *pred)

{
   float in[16];
   in[0] = (*I)(exampleNumber,0);
   in[1] = (*I)(exampleNumber,1);
   in[2] = (*I)(exampleNumber,2);
   in[3] = (*I)(exampleNumber,3);
   in[4] = (*I)(exampleNumber,4);
   in[5] = (*I)(exampleNumber,5);
   in[6] = (*I)(exampleNumber,6);
   in[7] = (*I)(exampleNumber,7);
   in[8] = (*I)(exampleNumber,8);
   in[9] = (*I)(exampleNumber,9);
   in[10] = (*I)(exampleNumber,10);
   in[11] = (*I)(exampleNumber,11);
   in[12] = (*I)(exampleNumber,12);
   in[13] = (*I)(exampleNumber,13);
   in[14] = (*I)(exampleNumber,14);
   in[15] = (*I)(exampleNumber,15);
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
   h1_0 += in[8] * p[90];
   h1_1 += in[8] * p[91];
   h1_2 += in[8] * p[92];
   h1_3 += in[8] * p[93];
   h1_4 += in[8] * p[94];
   h1_5 += in[8] * p[95];
   h1_6 += in[8] * p[96];
   h1_7 += in[8] * p[97];
   h1_8 += in[8] * p[98];
   h1_9 += in[8] * p[99];
   h1_0 += in[9] * p[100];
   h1_1 += in[9] * p[101];
   h1_2 += in[9] * p[102];
   h1_3 += in[9] * p[103];
   h1_4 += in[9] * p[104];
   h1_5 += in[9] * p[105];
   h1_6 += in[9] * p[106];
   h1_7 += in[9] * p[107];
   h1_8 += in[9] * p[108];
   h1_9 += in[9] * p[109];
   h1_0 += in[10] * p[110];
   h1_1 += in[10] * p[111];
   h1_2 += in[10] * p[112];
   h1_3 += in[10] * p[113];
   h1_4 += in[10] * p[114];
   h1_5 += in[10] * p[115];
   h1_6 += in[10] * p[116];
   h1_7 += in[10] * p[117];
   h1_8 += in[10] * p[118];
   h1_9 += in[10] * p[119];
   h1_0 += in[11] * p[120];
   h1_1 += in[11] * p[121];
   h1_2 += in[11] * p[122];
   h1_3 += in[11] * p[123];
   h1_4 += in[11] * p[124];
   h1_5 += in[11] * p[125];
   h1_6 += in[11] * p[126];
   h1_7 += in[11] * p[127];
   h1_8 += in[11] * p[128];
   h1_9 += in[11] * p[129];
   h1_0 += in[12] * p[130];
   h1_1 += in[12] * p[131];
   h1_2 += in[12] * p[132];
   h1_3 += in[12] * p[133];
   h1_4 += in[12] * p[134];
   h1_5 += in[12] * p[135];
   h1_6 += in[12] * p[136];
   h1_7 += in[12] * p[137];
   h1_8 += in[12] * p[138];
   h1_9 += in[12] * p[139];
   h1_0 += in[13] * p[140];
   h1_1 += in[13] * p[141];
   h1_2 += in[13] * p[142];
   h1_3 += in[13] * p[143];
   h1_4 += in[13] * p[144];
   h1_5 += in[13] * p[145];
   h1_6 += in[13] * p[146];
   h1_7 += in[13] * p[147];
   h1_8 += in[13] * p[148];
   h1_9 += in[13] * p[149];
   h1_0 += in[14] * p[150];
   h1_1 += in[14] * p[151];
   h1_2 += in[14] * p[152];
   h1_3 += in[14] * p[153];
   h1_4 += in[14] * p[154];
   h1_5 += in[14] * p[155];
   h1_6 += in[14] * p[156];
   h1_7 += in[14] * p[157];
   h1_8 += in[14] * p[158];
   h1_9 += in[14] * p[159];
   h1_0 += in[15] * p[160];
   h1_1 += in[15] * p[161];
   h1_2 += in[15] * p[162];
   h1_3 += in[15] * p[163];
   h1_4 += in[15] * p[164];
   h1_5 += in[15] * p[165];
   h1_6 += in[15] * p[166];
   h1_7 += in[15] * p[167];
   h1_8 += in[15] * p[168];
   h1_9 += in[15] * p[169];
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
   register float h2_0 = p[170];
   register float h2_1 = p[171];
   register float h2_2 = p[172];
   h2_0 += h1_0 * p[173];
   h2_1 += h1_0 * p[174];
   h2_2 += h1_0 * p[175];
   h2_0 += h1_1 * p[176];
   h2_1 += h1_1 * p[177];
   h2_2 += h1_1 * p[178];
   h2_0 += h1_2 * p[179];
   h2_1 += h1_2 * p[180];
   h2_2 += h1_2 * p[181];
   h2_0 += h1_3 * p[182];
   h2_1 += h1_3 * p[183];
   h2_2 += h1_3 * p[184];
   h2_0 += h1_4 * p[185];
   h2_1 += h1_4 * p[186];
   h2_2 += h1_4 * p[187];
   h2_0 += h1_5 * p[188];
   h2_1 += h1_5 * p[189];
   h2_2 += h1_5 * p[190];
   h2_0 += h1_6 * p[191];
   h2_1 += h1_6 * p[192];
   h2_2 += h1_6 * p[193];
   h2_0 += h1_7 * p[194];
   h2_1 += h1_7 * p[195];
   h2_2 += h1_7 * p[196];
   h2_0 += h1_8 * p[197];
   h2_1 += h1_8 * p[198];
   h2_2 += h1_8 * p[199];
   h2_0 += h1_9 * p[200];
   h2_1 += h1_9 * p[201];
   h2_2 += h1_9 * p[202];
   register float h3_0 = p[203];
   register float h3_1 = p[204];
   register float h3_2 = p[205];
   register float h3_3 = p[206];
   register float h3_4 = p[207];
   register float h3_5 = p[208];
   register float h3_6 = p[209];
   register float h3_7 = p[210];
   register float h3_8 = p[211];
   register float h3_9 = p[212];
   h3_0 += h2_0 * p[213];
   h3_1 += h2_0 * p[214];
   h3_2 += h2_0 * p[215];
   h3_3 += h2_0 * p[216];
   h3_4 += h2_0 * p[217];
   h3_5 += h2_0 * p[218];
   h3_6 += h2_0 * p[219];
   h3_7 += h2_0 * p[220];
   h3_8 += h2_0 * p[221];
   h3_9 += h2_0 * p[222];
   h3_0 += h2_1 * p[223];
   h3_1 += h2_1 * p[224];
   h3_2 += h2_1 * p[225];
   h3_3 += h2_1 * p[226];
   h3_4 += h2_1 * p[227];
   h3_5 += h2_1 * p[228];
   h3_6 += h2_1 * p[229];
   h3_7 += h2_1 * p[230];
   h3_8 += h2_1 * p[231];
   h3_9 += h2_1 * p[232];
   h3_0 += h2_2 * p[233];
   h3_1 += h2_2 * p[234];
   h3_2 += h2_2 * p[235];
   h3_3 += h2_2 * p[236];
   h3_4 += h2_2 * p[237];
   h3_5 += h2_2 * p[238];
   h3_6 += h2_2 * p[239];
   h3_7 += h2_2 * p[240];
   h3_8 += h2_2 * p[241];
   h3_9 += h2_2 * p[242];
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
   o = p[243];
   o += h3_0 * p[244];
   o += h3_1 * p[245];
   o += h3_2 * p[246];
   o += h3_3 * p[247];
   o += h3_4 * p[248];
   o += h3_5 * p[249];
   o += h3_6 * p[250];
   o += h3_7 * p[251];
   o += h3_8 * p[252];
   o += h3_9 * p[253];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,0) = o;
   }
   o -= in[0];
   sum += o*o;
   o = p[254];
   o += h3_0 * p[255];
   o += h3_1 * p[256];
   o += h3_2 * p[257];
   o += h3_3 * p[258];
   o += h3_4 * p[259];
   o += h3_5 * p[260];
   o += h3_6 * p[261];
   o += h3_7 * p[262];
   o += h3_8 * p[263];
   o += h3_9 * p[264];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,1) = o;
   }
   o -= in[1];
   sum += o*o;
   o = p[265];
   o += h3_0 * p[266];
   o += h3_1 * p[267];
   o += h3_2 * p[268];
   o += h3_3 * p[269];
   o += h3_4 * p[270];
   o += h3_5 * p[271];
   o += h3_6 * p[272];
   o += h3_7 * p[273];
   o += h3_8 * p[274];
   o += h3_9 * p[275];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,2) = o;
   }
   o -= in[2];
   sum += o*o;
   o = p[276];
   o += h3_0 * p[277];
   o += h3_1 * p[278];
   o += h3_2 * p[279];
   o += h3_3 * p[280];
   o += h3_4 * p[281];
   o += h3_5 * p[282];
   o += h3_6 * p[283];
   o += h3_7 * p[284];
   o += h3_8 * p[285];
   o += h3_9 * p[286];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,3) = o;
   }
   o -= in[3];
   sum += o*o;
   o = p[287];
   o += h3_0 * p[288];
   o += h3_1 * p[289];
   o += h3_2 * p[290];
   o += h3_3 * p[291];
   o += h3_4 * p[292];
   o += h3_5 * p[293];
   o += h3_6 * p[294];
   o += h3_7 * p[295];
   o += h3_8 * p[296];
   o += h3_9 * p[297];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,4) = o;
   }
   o -= in[4];
   sum += o*o;
   o = p[298];
   o += h3_0 * p[299];
   o += h3_1 * p[300];
   o += h3_2 * p[301];
   o += h3_3 * p[302];
   o += h3_4 * p[303];
   o += h3_5 * p[304];
   o += h3_6 * p[305];
   o += h3_7 * p[306];
   o += h3_8 * p[307];
   o += h3_9 * p[308];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,5) = o;
   }
   o -= in[5];
   sum += o*o;
   o = p[309];
   o += h3_0 * p[310];
   o += h3_1 * p[311];
   o += h3_2 * p[312];
   o += h3_3 * p[313];
   o += h3_4 * p[314];
   o += h3_5 * p[315];
   o += h3_6 * p[316];
   o += h3_7 * p[317];
   o += h3_8 * p[318];
   o += h3_9 * p[319];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,6) = o;
   }
   o -= in[6];
   sum += o*o;
   o = p[320];
   o += h3_0 * p[321];
   o += h3_1 * p[322];
   o += h3_2 * p[323];
   o += h3_3 * p[324];
   o += h3_4 * p[325];
   o += h3_5 * p[326];
   o += h3_6 * p[327];
   o += h3_7 * p[328];
   o += h3_8 * p[329];
   o += h3_9 * p[330];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,7) = o;
   }
   o -= in[7];
   sum += o*o;
   o = p[331];
   o += h3_0 * p[332];
   o += h3_1 * p[333];
   o += h3_2 * p[334];
   o += h3_3 * p[335];
   o += h3_4 * p[336];
   o += h3_5 * p[337];
   o += h3_6 * p[338];
   o += h3_7 * p[339];
   o += h3_8 * p[340];
   o += h3_9 * p[341];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,8) = o;
   }
   o -= in[8];
   sum += o*o;
   o = p[342];
   o += h3_0 * p[343];
   o += h3_1 * p[344];
   o += h3_2 * p[345];
   o += h3_3 * p[346];
   o += h3_4 * p[347];
   o += h3_5 * p[348];
   o += h3_6 * p[349];
   o += h3_7 * p[350];
   o += h3_8 * p[351];
   o += h3_9 * p[352];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,9) = o;
   }
   o -= in[9];
   sum += o*o;
   o = p[353];
   o += h3_0 * p[354];
   o += h3_1 * p[355];
   o += h3_2 * p[356];
   o += h3_3 * p[357];
   o += h3_4 * p[358];
   o += h3_5 * p[359];
   o += h3_6 * p[360];
   o += h3_7 * p[361];
   o += h3_8 * p[362];
   o += h3_9 * p[363];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,10) = o;
   }
   o -= in[10];
   sum += o*o;
   o = p[364];
   o += h3_0 * p[365];
   o += h3_1 * p[366];
   o += h3_2 * p[367];
   o += h3_3 * p[368];
   o += h3_4 * p[369];
   o += h3_5 * p[370];
   o += h3_6 * p[371];
   o += h3_7 * p[372];
   o += h3_8 * p[373];
   o += h3_9 * p[374];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,11) = o;
   }
   o -= in[11];
   sum += o*o;
   o = p[375];
   o += h3_0 * p[376];
   o += h3_1 * p[377];
   o += h3_2 * p[378];
   o += h3_3 * p[379];
   o += h3_4 * p[380];
   o += h3_5 * p[381];
   o += h3_6 * p[382];
   o += h3_7 * p[383];
   o += h3_8 * p[384];
   o += h3_9 * p[385];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,12) = o;
   }
   o -= in[12];
   sum += o*o;
   o = p[386];
   o += h3_0 * p[387];
   o += h3_1 * p[388];
   o += h3_2 * p[389];
   o += h3_3 * p[390];
   o += h3_4 * p[391];
   o += h3_5 * p[392];
   o += h3_6 * p[393];
   o += h3_7 * p[394];
   o += h3_8 * p[395];
   o += h3_9 * p[396];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,13) = o;
   }
   o -= in[13];
   sum += o*o;
   o = p[397];
   o += h3_0 * p[398];
   o += h3_1 * p[399];
   o += h3_2 * p[400];
   o += h3_3 * p[401];
   o += h3_4 * p[402];
   o += h3_5 * p[403];
   o += h3_6 * p[404];
   o += h3_7 * p[405];
   o += h3_8 * p[406];
   o += h3_9 * p[407];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,14) = o;
   }
   o -= in[14];
   sum += o*o;
   o = p[408];
   o += h3_0 * p[409];
   o += h3_1 * p[410];
   o += h3_2 * p[411];
   o += h3_3 * p[412];
   o += h3_4 * p[413];
   o += h3_5 * p[414];
   o += h3_6 * p[415];
   o += h3_7 * p[416];
   o += h3_8 * p[417];
   o += h3_9 * p[418];
   if(IS_PRED == true) {
      (*pred)(exampleNumber,15) = o;
      return 0.;
   }
   o -= in[15];
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


