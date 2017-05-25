#include <iostream>
using namespace std;

#include "DataType.hpp"

#include "generatedFunc.hpp"
#include "ObjFcn.hpp"
#include <algorithm>
#include <nlopt.h>
#include "util.hpp"

extern double getTime();

int main(int argc, char* argv[])
{

  if(argc < 3) {
    fprintf(stderr,"Use: datafile paramFile\n");
    return -1;
  }
  // load the data and parameters
  ObjFuncVec<DATA_TYPE, struct generatedFcnInterest<DATA_TYPE> > *oFuncVec =
    init< DATA_TYPE, struct generatedFcnInterest<DATA_TYPE> > (argv[1], argv[2],true);

  ObjFcn<DATA_TYPE, struct generatedFcnInterest<DATA_TYPE> > *oFunc= oFuncVec->vec[0];
  assert(oFunc->devID < 0);

  //cout << "Objective Function: " << oFunc->name() << endl
       //<< "Function of Interest: " << oFunc->FcnInterest_name()
       //<< " with G() " << oFunc->FcnInterest_gFcnName() << endl;

  oFunc->offloadParam(oFuncVec->param);

  uint32_t nExamples = oFunc->InputExample().rows();
  uint32_t nOutput = oFunc->KnownExample().cols();
  uint32_t nInput = oFunc->InputExample().cols();
  if(nOutput == 0) // have autoencoder
    nOutput = oFunc->InputExample().cols();
  Matrix<DATA_TYPE> Pred(nExamples, nOutput);
  oFunc->pred(&Pred);

  // output in CSV format
  for(int i=0; i < nExamples; i++) {
    if(oFunc->KnownExample().cols() == 0) { // autoencoder special case
      std::cout << "pred";
      for(int j=0; j < nInput; j++) std::cout << ", " << Pred(i,j);
      std::cout << ", known"; 
      for(int j=0; j < nInput; j++) std::cout << ", " << oFunc->InputExample(i,j);
      std::cout << endl;
    } else {
      std::cout << "pred"; 
      for(int j=0; j < nOutput; j++) std::cout << ", " << Pred(i,j);
      std::cout << ", known"; 
      for(int j=0; j < nOutput; j++) std::cout << ", " << oFunc->KnownExample(i,j);
      std::cout << endl;
    }
  }

  return 0;
}
