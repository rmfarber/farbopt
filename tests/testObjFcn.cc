#include <iostream>
using namespace std;
#include "xor.hpp"
#include "ObjFcn.hpp"
#include "util.hpp"

#include <nlopt.h>

extern "C" double nloptFunc(unsigned int n, const double *x, double *grad, void* f_data)
{
  ObjFcn<float, generatedFcnInterest<float> > *oFunc = (ObjFcn<float, generatedFcnInterest<float> > *) f_data;
  assert(n == oFunc->nParam());
#pragma SIMD
  for(int i=0; i < n; ++i)
    oFunc->Param()[i] = x[i];

  return oFunc->func();
}


int main(int argc, char* argv[])
{
  ObjFcn<float, struct generatedFcnInterest<float> > LMS;

  cout << "Objective Function: " << LMS.name() << endl
       << "Function of Interest: " << LMS.FcnInterest_name()
       << " with G() " << LMS.FcnInterest_gFcnName() << endl;

  if(argc < 3) {
    fprintf(stderr,"Use: datafile paramFile\n");
    return -1;
  }
  cout << "Number Parameters " << LMS.nParam() << endl;
  
  initData("-", LMS.InputExample(), LMS.KnownExample());

  nlopt_opt opt = nlopt_create(NLOPT_LN_PRAXIS, LMS.nParam()); // algorithm and dimensionality
  // NOTE: alternative optimization methods ...
  //opt = nlopt_create(NLOPT_LN_NEWUOA, LMS.nParam());
  //opt = nlopt_create(NLOPT_LN_COBYLA, LMS.nParam());
  //opt = nlopt_create(NLOPT_LN_BOBYQA, LMS.nParam());
  //opt = nlopt_create(NLOPT_LN_AUGLAG, LMS.nParam());

  nlopt_set_min_objective(opt, objFunc, (void*) &LMS);
#if defined(MAX_RUNTIME) 
  fprintf(stderr,"Warning: performing a quick %d second test!\n", MAX_RUNTIME);
  nlopt_set_maxtime(opt, MAX_RUNTIME); // Use for running quick tests
#else
  fprintf(stderr,"MAX runtime %d seconds!\n", 120*60);
  nlopt_set_maxtime(opt, (120. * 60.)); // maximum runtime in seconds
#endif


  int n=LMS.nParam();
  cerr << n << endl;
  double *x = new double[n];
  cerr << nloptFunc(LMS.nParam(), x, NULL, (void*) &LMS) << endl;
  cerr << LMS.func() << endl;
  delete [] x;
  
  return 0;
}
