#include <iostream>
using namespace std;
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
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
    if(cbufIndex>=maxSize) cbufIndex=0;
    return(f);
  }

  const float& operator[](const int i) const {
    int index=i+cbufIndex;
    return(val[index % val.size()]);
  }
  inline int size() { return(val.size()); }
};

  
#ifdef TEST_CIRCULAR_BUFFER
int main(int argc, char *argv[])
{
  int delta=6;
  int m=4+1;// input + output
  cBufferIndex cbuf(m*delta);
  for(int i=0; i < cbuf.size(); i++) cbuf.push(i);
  for(int i=0; i < 20; i++) {
    cout << "*********" << endl;
    for(int j=0; j < 4; j++) {
      int index=i + delta*j;
      cout << index << " " << cbuf[index] << endl;
    }
  }
}
#else

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
  for(int i=0; i< 10*bufsize; i++) MackeyGlass(i, y, b, c, tau);
  
  // write header info
  uint32_t nInput=m+1; fwrite(&nInput,sizeof(int32_t), 1, fn);
  uint32_t nOutput=1; fwrite(&nOutput,sizeof(int32_t), 1, fn);
  uint32_t nExamples=nVec; fwrite(&nExamples,sizeof(int32_t), 1, fn);

  for(int i=0; i < nVec; i++) {
    float yOut[nInput];
    MackeyGlass(i, y, b, c, tau);
    for(int j=0; j < nInput; j++) {
      int index = (j+1)*delta; // push based circular buffer, so tau is index into the past
      yOut[j] = y[index];
    }
#define BINARY_OUT
#ifdef BINARY_OUT
    fwrite(yOut, sizeof(float), nInput, fn);
    float output=y[0];
    fwrite(&output, sizeof(float), 1, fn);
#else
    for(int j=0; j < nInput; j++)  cout << yOut[j] << " ";
    cout << y[0] << endl;
#endif
  }
}

int main(int argc, char *argv[])
{
  if(argc < 6) {
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

#endif
