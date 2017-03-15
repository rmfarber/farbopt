#include <iostream>
using namespace std;
#include "stdint.h"
#include "Matrix.hpp"

int main()
{
  int nExamples=10;
  int nInput=15;
  Matrix<float> m(nExamples,nInput);

  double err=0.;
#pragma acc parallel loop present(m) reduction(+:err)
  for(int i=0; i < nExamples; i++) {
    float x=i;
    for(int j=0; j < nInput; j++) {
      x += 0.1;
      m(i,j) = x;
      err += x;
    }
  }
  cerr << "Err " << err << endl;
float *pt = *m.getDataPtrAddr();
#pragma acc update host(pt[0:(nInput*nExamples)])

  for(int i=0; i < nExamples; i++) {
    for(int j=0; j < nInput; j++) {
      cout << m(i,j) << ' '; 
    }
    cout << endl;
  }
  
  return 0;
}
