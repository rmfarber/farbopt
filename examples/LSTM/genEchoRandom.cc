#include <iostream>
using namespace std;

#include <cstring>
#include "FcnOfInterest_config.h"

int main(int argc, char *argv[])
{
  char *filename=argv[1];
  FILE *fn=stdout;
  float inc=0.01;

  if(strcmp("-", filename) != 0)
    fn=fopen(filename,"w");

  if(!fn) {
    fprintf(stderr,"Cannot open %s\n",filename);
    exit(1);
  }

  uint32_t nExamples=atoi(argv[2]);
  uint32_t srand(atoi(argv[3]));

  // write header info
  uint32_t nInput=N_INPUT; fwrite(&nInput,sizeof(int32_t), 1, fn);
  uint32_t nOutput=N_OUTPUT; fwrite(&nOutput,sizeof(int32_t), 1, fn);
  fwrite(&nExamples,sizeof(int32_t), 1, fn);

  float buf[N_INPUT];
  float start=0.;
  for(uint32_t seq=0; seq < nExamples; seq++) {
    for(int i=0; i < N_INPUT; i++) buf[i] = ((int)(100.*rand()/((float)RAND_MAX)))/100.;
    //for(int i=0; i < N_INPUT; i++) cout << buf[i] << " "; cout << endl;
    fwrite(buf, sizeof(float), N_INPUT, fn);
    fwrite(buf, sizeof(float), N_INPUT, fn);
    start += inc;
  }
}
