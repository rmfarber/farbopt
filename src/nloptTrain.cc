#include <iostream>
//using namespace std;

#include "DataType.hpp"

#include "generatedFunc.hpp"
#include "ObjFcn.hpp"
#include <vector>
#include <nlopt.h>
#include "util.hpp"
#include <algorithm>

extern double getTime();
extern double nloptFunc(unsigned int n, const double *x, double *grad, void* f_data);

int main(int argc, char* argv[])
{
  uint32_t maxRuntime = 120*60;
  if(argc < 3) {
    fprintf(stderr,"Use: datafile paramFile MaxRuntime (seconds)\n");
    return -1;
  }

  if(argc > 3) { maxRuntime = atoi(argv[3]); }
  cout << "Max Runtime is " << maxRuntime << " (seconds)" << endl;
  
  ObjFuncVec<DATA_TYPE, struct generatedFcnInterest<DATA_TYPE> > *oFuncVec =
    init< DATA_TYPE, struct generatedFcnInterest<DATA_TYPE> > (argv[1], argv[2],false);

  // algorithm and dimensionality
  int nParam = oFuncVec->nParam;
  nlopt_opt opt = nlopt_create(NLOPT_LN_PRAXIS, nParam);
  printf("Using NLOPT_LN_PRAXIS\n");

  // NOTE: alternative optimization methods ...
  //opt = nlopt_create(NLOPT_LN_NEWUOA, oFunc->nParam());
  //printf("Using NLOPT_LN_NEWUOA\n");
  //opt = nlopt_create(NLOPT_LN_COBYLA, oFunc->nParam());
  //printf("Using NLOPT_LN_COBYLA\n");
  //opt = nlopt_create(NLOPT_LN_BOBYQA, oFunc->nParam());
  //printf("Using NLOPT_LN_BOBYQA\n");
  //opt = nlopt_create(NLOPT_LN_AUGLAG, oFunc->nParam());
  //printf("Using NLOPT_LN_AUGLAG\n");

  nlopt_set_min_objective(opt, nloptFunc, (void*) oFuncVec);
  nlopt_set_maxtime(opt,maxRuntime);

  vector<double> x(nParam);
  double minf; /* the minimum objective value, upon return */

  for(int i=0; i < nParam; i++) x[i] = oFuncVec->param[i];

  double startTime=getTime();
  int ret=nlopt_optimize(opt, &x[0], &minf);

  cerr << "\tOptimization Time " << (getTime()-startTime) << endl;
  if (ret < 0) {
    printf("\tnlopt failed! ret %d\n", ret);
  } else {
    printf("\tfound minimum %0.10g ret %d\n", minf,ret);
  }
  
  fini< DATA_TYPE, struct generatedFcnInterest<DATA_TYPE> >(argv[2], oFuncVec);
  nlopt_destroy(opt);

  return 0;
}
