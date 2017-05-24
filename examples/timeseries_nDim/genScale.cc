#include <iostream>
using namespace std;
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <vector>

int main(int argc, char *argv[])
{
  if(argc < 1) {
    fprintf(stderr,"Use: filename");
    exit(1);
  }
  char *filename=argv[1];
  FILE *fin=stdin;

  if(strcmp("-", filename) != 0)
    fin=fopen(filename,"r");

  if(!fin) {
    fprintf(stderr,"Cannot open %s\n",filename);
    exit(1);
  }
  
  // read header info
  uint32_t nInput; fread(&nInput,sizeof(int32_t), 1, fin);
  uint32_t nOutput; fread(&nOutput,sizeof(int32_t), 1, fin);
  uint32_t nExamples; fread(&nExamples,sizeof(int32_t), 1, fin);
  float yOut[nInput+nOutput];
  bool beenHere=false;
  float minVal=0., maxVal=0.;
  cerr << "nInput " << nInput << endl;
  cerr << "nOutput " << nOutput << endl;
  cerr << "nExamples " << nExamples << endl;
  for(int i=0; i < nExamples; i++) {
    fread(yOut, sizeof(float), (nInput+nOutput), fin);
    if(!beenHere) {minVal=maxVal=yOut[0]; beenHere=true;}
    
    for(int j=0; j < nInput+nOutput; j++) {
      minVal=min(yOut[j],minVal); maxVal=max(maxVal, yOut[j]);
    }
    
  }
  cout << "#define MAX_VAL " << maxVal << endl;
  cout << "#define MIN_VAL " << minVal << endl;
  return 0;
}
