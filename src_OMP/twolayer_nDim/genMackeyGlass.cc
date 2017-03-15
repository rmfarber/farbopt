#include <iostream>
using namespace std;
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <vector>

// get a uniform random number between -1 and 1 
inline float f_rand() {
  return 2*(rand()/((float)RAND_MAX)) -1.;
}

inline int cBufferIndex(int index, int maxSize, int offset)
{
  int n = (index%maxSize);
  int next = n + offset;
  if(offset < 0) {
    if(next < 0) next = maxSize + n+offset;
  } else if(offset > 0) {
    if(next >= maxSize) next -= maxSize;
  }
  return next;
}

void MackeyGlass(int index, vector<double> &y, double b, double c, int tau)
{
  int t = cBufferIndex(index,y.size(), 0);
  int t_plus_1 = cBufferIndex(index,y.size(), 1);
  int t_minus_tau = cBufferIndex(index,y.size(), -tau);
  
  y[t_plus_1] = y[t] - b*y[t] + c*y[t_minus_tau]/(1.+pow(y[t_minus_tau],10.));
}

void genData(FILE *fn, int nVec, double b, double c, int tau, int nDim)
{
  int bufsize=tau+1;
  vector<double> y;
  y.resize(bufsize);

  //initialized y
  for(int i=0; i < y.size(); i++)
    y[i] = abs(f_rand());
    
  // fall down to the attractor
  for(int i=0; i< bufsize; i++) {
   MackeyGlass(i, y, b, c, tau);
  }
  
#define BINARY_OUT
#ifdef BINARY_OUT
  // write header info
  uint32_t nInput=nDim; fwrite(&nInput,sizeof(int32_t), 1, fn);
  uint32_t nOutput=1; fwrite(&nOutput,sizeof(int32_t), 1, fn);
  uint32_t nExamples=nVec; fwrite(&nExamples,sizeof(int32_t), 1, fn);

  for(int i=0; i < nVec; i++) {
    float yOut[nInput+nOutput];
    MackeyGlass(i, y, b, c, tau);
    for(int j=0; j < (nInput+nOutput); j++) {
      yOut[j] = y[cBufferIndex((i+1)-j,y.size(),tau)];
    }
    fwrite(yOut, sizeof(float), (nInput+nOutput), fn);
  }
#else
  for(int i=0; i < nVec; i++) {
    MackeyGlass(i, y, b, c, tau);
    fprintf(fn, "%d, %g\n",i,
	    y[cBufferIndex(i+1,y.size(),-tau)]);
  }
#endif
}

int main(int argc, char *argv[])
{
  if(argc < 5) {
    fprintf(stderr,"Use: filename nVec b(0.1) c(0.2) tau(17) nDim seed\n");
    exit(1);
  }
  char *filename=argv[1];
  FILE *fn=stdout;

  if(strcmp("-", filename) != 0)
    fn=fopen(filename,"w");

  if(!fn) {
    fprintf(stderr,"Cannot open %s\n",filename);
    exit(1);
  }
  int nVec = atoi(argv[2]);
  double b = atof(argv[3]);
  double c = atof(argv[4]);
  int tau = atoi(argv[5]);
  int nDim = atoi(argv[6]);

  srand(atoi(argv[7]));
  genData(fn, nVec, b, c, tau, nDim);

  if(fn != stdout) fclose(fn);
  return 0;
}
