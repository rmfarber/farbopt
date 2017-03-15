#include <iostream>
using namespace std;
#include "stdint.h"
#include "Matrix.hpp"

template< typename REAL_T>
class Test {
public:
  int foo;
  Matrix<float> Input, Known;

  Test(int nExamples, int colsInput, int colsKnown)
  {
    Input.reserve(nExamples, colsInput );
    Known.reserve(nExamples, colsKnown );
    cerr << "before copyin size " << sizeof(*this) <<  endl; 

   REAL_T **pInput=Input.getDataPtrAddr();
	cerr << "Input.data_ " << Input.data_ << " " << pInput << endl;
#pragma acc enter data copyin(this)
#pragma acc enter data create(pInput[0:nExamples*colsInput])
    cerr << "did copyin" << endl;
  }
  FCN_ATTRIBUTES
  inline REAL_T& InputExample(uint32_t i, uint32_t j) { return (Input(i,j)); }
  FCN_ATTRIBUTES
  inline REAL_T& KnownExample(uint32_t i, uint32_t j) { return (Known(i,j)); }
  FCN_ATTRIBUTES
  inline Matrix<REAL_T>& InputExample() { return Input; }
  FCN_ATTRIBUTES
  inline Matrix<REAL_T>& KnownExample() { return Known; }
};

int main()
{
  int nExamples=10;
  int nInput=2;
  int nKnown=1;
  //Matrix<float> m(nExamples,nInput);
  Test<float> oFunc(nExamples, nInput, nKnown);

  double err=0.;
#pragma acc parallel loop reduction(+:err)
  for(int i=0; i < nExamples; i++) {
    float x=i;
    for(int j=0; j < nInput; j++) {
      x += 0.1;
      oFunc.InputExample(i,j) = x;
      err += x;
    }
  }
  cerr << "Err " << err << endl;
 float *pt = *oFunc.Input.getDataPtrAddr();
#pragma acc update host(pt[0:(nInput*nExamples)])

  for(int i=0; i < nExamples; i++) {
    for(int j=0; j < nInput; j++) {
      cout << oFunc.InputExample(i,j) << ' '; 
    }
    cout << endl;
  }
  return 0;
}
