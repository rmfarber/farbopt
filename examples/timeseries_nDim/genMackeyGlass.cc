#include <iostream>
using namespace std;
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <cmath>
#include <vector>

struct cBufferIndex {
private:
  int cbufIndex;
  int maxSize;
  vector<float> val;
  cBufferIndex() {}

public:
  cBufferIndex(int maxSize) : maxSize(maxSize), cbufIndex(0)
  {
    val.resize(maxSize);
  }

  inline float push(float f) {
    val[cbufIndex++] = f;
    if(cbufIndex >= val.size() ) cbufIndex=0;
    return(f);
  }

  const float& operator[](const int i) const {
    int index= (cbufIndex-1) - i;
    if(index < 0) index = val.size() - abs(index);
    if(index < 0) {
      cerr << "Index " << i << " exceeded circular buffer history" << endl;
      exit(1);
    }
    assert(index >=0);
    assert(index < val.size());
    return(val[index]);
  }
  inline int size() { return(val.size()); }
};

// get a uniform random number between -1 and 1 
inline float f_rand() {
  return 2*(rand()/((float)RAND_MAX)) -1.;
}

// this uses vector_y as a circular buffer
void MackeyGlass(int index, cBufferIndex &y, double b, double c, int tau)
{
  // push based circular buffer, so tau is index into the past
  float y_t_plus_1 = y[0] - b*y[0] + c*y[tau]/(1.+pow(y[tau],10.));
  //static int static_index=0;
  //float y_t_plus_1 = static_index++;
  y.push(y_t_plus_1);
}

void genData(FILE *fn, int nVec, double b, double c, int tau, int m, int delta)
{
  int bufsize=m*delta*tau+1;
  cBufferIndex y(bufsize);

  //initialized y to random values
  for(int i=0; i < y.size(); i++) y.push(f_rand());
    
  // fall down to the attractor
  for(int i=0; i< bufsize; i++) MackeyGlass(i, y, b, c, tau);
  
  // write header info
  uint32_t nInput=m+1; fwrite(&nInput,sizeof(int32_t), 1, fn);
  uint32_t nOutput=1; fwrite(&nOutput,sizeof(int32_t), 1, fn);
  uint32_t nExamples=nVec; fwrite(&nExamples,sizeof(int32_t), 1, fn);

  for(int i=0; i < nVec; i++) {
    int n = nInput+nOutput;
    float yOut[n];
    MackeyGlass(i, y, b, c, tau);
    for(int j=0; j < n; j++) {
      int index = j*delta; // push based circular buffer, so tau is index into the past
      yOut[j] = y[index];
    }
#define BINARY_OUT
#ifdef BINARY_OUT
    fwrite(&yOut[1], sizeof(float), nInput, fn);
    float output=y[0];
    fwrite(&output, sizeof(float), 1, fn);
#else
    for(int j=1; j <= nInput; j++)  cout << yOut[j] << " ";
    cout << yOut[0] << endl;
#endif
  }
}

int main(int argc, char *argv[])
{
  if(argc < 9) {
    fprintf(stderr,"Use: filename nVec b(0.1) c(0.2) tau(17) m(3) Delta(6) seed\n");
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
  int m = atoi(argv[6]);
  int delta = atoi(argv[7]);

  srand(atoi(argv[7]));
  genData(fn, nVec, b, c, tau, m, delta);

  if(fn != stdout) fclose(fn);
  return 0;
}

