#include <iostream>
using namespace std;
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <cmath>
#include <vector>
#include "scale.hpp"

#define NDEBUG

inline float rescale(float val, float minVal, float maxVal, float minRange, float maxRange)
{
  val = (maxRange-minRange)*((val-minVal)/(maxVal-minVal)) + minRange; // minRange to maxRange
  assert(val >= minRange && val <= maxRange);
  return(val);
}

int main(int argc, char *argv[])
{
  if(argc < 2) {
    cerr << "Use: minRange maxRange" << endl;
    exit(0);
  }
  float minRange=atof(argv[1]);
  float maxRange=atof(argv[2]);

  FILE *fin=stdin;
  FILE *fout=stdout;
  // read header info
  uint32_t nInput; fread(&nInput,sizeof(int32_t), 1, fin);
  uint32_t nOutput; fread(&nOutput,sizeof(int32_t), 1, fin);
  uint32_t nExamples; fread(&nExamples,sizeof(int32_t), 1, fin);
  float yOut[nInput+nOutput];

  fwrite(&nInput,sizeof(int32_t), 1, fout);
  fwrite(&nOutput,sizeof(int32_t), 1, fout);
  fwrite(&nExamples,sizeof(int32_t), 1, fout);

  for(int i=0; i < nExamples; i++) {
    fread(yOut, sizeof(float), (nInput+nOutput), fin);
    
#pragma omp parallel for
    for(int j=0; j < nInput+nOutput; j++) {
      yOut[j] = rescale(yOut[j], MIN_VAL, MAX_VAL, minRange, maxRange);
    }
    fwrite(yOut, sizeof(float), (nInput+nOutput), fout);
  }
  return 0;
}
