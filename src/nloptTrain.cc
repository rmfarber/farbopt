#include <iostream>
//using namespace std;

#include "DataType.hpp"

#include "FcnOfInterest.hpp"
#include "ObjFcn.hpp"
#include <vector>
#include <nlopt.h>
#include "util.hpp"
#include <algorithm>
#include <getopt.h>

extern double getTime();
extern double nloptFunc(unsigned int n, const double *x, double *grad, void* f_data);

void PrintHelp(char *PgmName)
{
  cout <<
    "************************** " << PgmName  << " ********************************\n"
    "--max_runtime <integer>:\tThe maximum seconds before stopping the optimization. default: 1 minute\n"
    "--checkpoint <integer>:\tTime in second between checkpoints. default:0 or no checkpoint\n"
    "--data <string>:\tname of the data file. REQUIRED\n"
    "--param <string>:\tname of the param file. REQUIRED\n"
    "--opt <string>:\tOptimization method. LD_ means a gradient is required\n"
    "\tLD_BFGS: Low-storage BFFGS method. DEFAULT  \n"
    "\tLD_SLSQP: Sequential Least-Squares Quadratic Programming. \n"
    "\tLD_VAR1: Shifted limited-memroy variable-metric using a rank 1 method. \n"
    "\tLD_VAR2: Shifted limited-memroy variable-metric using a rank 2 method. \n"
    "\tLD_MMA: Method of Moving Asymptotes and CCSA. \n"
    "\tLD_TNEWTON_PRECOND_RESTART: Preconditioned truncated Newton,\n"
    "\t\tpreconditioned by the low-storage BFGS algorithm with steepest-descent restarting. \n"
    "\tLD_TNEWTON_PRECOND: Same w/o restarting.\n"
    "\tLD_TNEWTON_RESTART: Same w/o preconditioning.\n"
    "\tLD_TNEWTON: Same w/o preconditioning or restarting.\n"
    "\tLN_BOBYQA: Constructs a quadratic approximation of the objective. Largely supercedes by NEWUOA. \n"
    "\tLN_PRAXIS: \n"
    "\tLN_NEWUOA: \n"
    "\tLN_COBYLA: \n"
    "\tLN_AUGLAG: \n"
    "--help:\tShow help\n";
  exit(1);
}

// global commandline values;
string datafile, paramfile;
uint32_t maxRuntime=60;
uint32_t checkPointInterval=0;

enum OptEnum {LD_BFGS, LD_SLSQP, LD_VAR1, LD_VAR2, LD_MMA,
	      LD_TNEWTON_PRECOND_RESTART, LD_TNEWTON_PRECOND, LD_TNEWTON_RESTART, LD_TNEWTON,
	      LN_BOBYQA, LN_PRAXIS, LN_NEWUOA, LN_COBYLA, LN_AUGLAG};
int OptMethod=OptEnum::LD_BFGS;
int ProcessArgs(int argc, char *argv[])
{
  if(argc == 1) {
    PrintHelp(argv[0]);
    return -1;
  }
  const char* const short_opts = "d:p:t:c:h";
  const option long_opts[] = {
    {"data", required_argument, nullptr, 'd'},
    {"param", required_argument, nullptr, 'p'},
    {"LD_BFGS", no_argument, &OptMethod, OptEnum::LD_BFGS},
    {"LD_SLSQP", no_argument, &OptMethod, OptEnum::LD_SLSQP},
    {"LD_VAR1", no_argument, &OptMethod, OptEnum::LD_VAR1},
    {"LD_VAR2", no_argument, &OptMethod, OptEnum::LD_VAR2},
    {"LD_MMA", no_argument, &OptMethod, OptEnum::LD_MMA},
    {"LD_TNEWTON_PRECOND_RESTART", no_argument, &OptMethod, OptEnum::LD_TNEWTON_PRECOND_RESTART},
    {"LD_TNEWTON_PRECOND", no_argument, &OptMethod, OptEnum::LD_TNEWTON_PRECOND},
    {"LD_TNEWTON_RESTART", no_argument, &OptMethod, OptEnum::LD_TNEWTON_RESTART},
    {"LD_TNEWTON", no_argument, &OptMethod, OptEnum::LD_TNEWTON},
    {"LN_BOBYQA", no_argument, &OptMethod, OptEnum::LN_BOBYQA},
    {"LN_PRAXIS", no_argument, &OptMethod, OptEnum::LN_PRAXIS},
    {"LN_NEWUOA", no_argument, &OptMethod, OptEnum::LN_NEWUOA},
    {"LN_COBYLA", no_argument, &OptMethod, OptEnum::LN_COBYLA},
    {"LN_AUGLAG", no_argument, &OptMethod, OptEnum::LN_AUGLAG},
    {"help", no_argument, nullptr, 'h'},
    {nullptr, no_argument, nullptr, 0}
  };
  while(true) {
    const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);
    if(-1 == opt)
      break;

    switch(opt) {
    case 'c': checkPointInterval = stoi(optarg); break;
    case 't': maxRuntime = stoi(optarg); break;
    case 'd': datafile = optarg; break;
    case 'p': paramfile = optarg; break;
    case 'h':
    case '?':
      PrintHelp(argv[0]);
      break;
    }
  }
  return(0);
}

int main(int argc, char* argv[])
{
  if(ProcessArgs(argc,argv) <0)
    return -1;

  if(datafile.empty()) {
    cout << "Fatal: must specify a training set" << endl;
    return -1;
  }
  cout << "training data in: " << datafile << endl;
  if(paramfile.empty()) {
    cout << "Fatal: must specify a parameter file " << endl;
    return -1;
  }
  cout << "using and/or writing params to: " << paramfile << endl;
  
  ObjFuncVec<DATA_TYPE, struct generatedFcnInterest<DATA_TYPE> > *oFuncVec =
    init< DATA_TYPE, struct generatedFcnInterest<DATA_TYPE> > (datafile.c_str(), paramfile.c_str(),false);

  cout << "Max Runtime is " << maxRuntime << " seconds" << endl;

  if(checkPointInterval > 0) {
    oFuncVec->setCheckPointInterval(checkPointInterval); 
    cout << "checkpoint interval is " << checkPointInterval << " seconds" << endl;
  }

  // algorithm and dimensionality
  int nParam = oFuncVec->nParam;

  nlopt_opt opt;
  switch (OptMethod) {
  case OptEnum::LN_BOBYQA:
    opt = nlopt_create(NLOPT_LN_BOBYQA, nParam);
    printf("Using NLOPT_LN_BOBYQA\n");
    break;
  case OptEnum::LD_BFGS:
    opt = nlopt_create(NLOPT_LD_LBFGS, nParam);
    printf("Using NLOPT_LD_LBFGS\n");
    break;
  case OptEnum::LD_SLSQP:
    opt = nlopt_create(NLOPT_LD_SLSQP, nParam);
    printf("Using NLOPT_LD_SLSQP\n");
    break;
  case OptEnum::LD_VAR1:
    opt = nlopt_create(NLOPT_LD_VAR1, nParam);
    printf("Using NLOPT_LD_VAR1\n");
    break;
  case OptEnum::LD_VAR2:
    opt = nlopt_create(NLOPT_LD_VAR2, nParam);
    printf("Using NLOPT_LD_VAR2\n");
    break;
  case OptEnum::LD_MMA:
    opt = nlopt_create(NLOPT_LD_MMA, nParam);
    printf("Using NLOPT_LD_MMA\n");
    break;
  case OptEnum::LD_TNEWTON_PRECOND_RESTART:
    opt = nlopt_create(NLOPT_LD_TNEWTON_PRECOND_RESTART, nParam);
    printf("Using NLOPT_LD_TNEWTON_PRECOND_RESTART\n");
    break;
  case OptEnum::LD_TNEWTON_PRECOND:
    opt = nlopt_create(NLOPT_LD_TNEWTON_PRECOND, nParam);
    printf("Using NLOPT_LD_TNEWTON_PRECOND\n");
    break;
  case OptEnum::LD_TNEWTON_RESTART:
    opt = nlopt_create(NLOPT_LD_TNEWTON_RESTART, nParam);
    printf("Using NLOPT_LD_TNEWTON_RESTART\n");
    break;
  case OptEnum::LD_TNEWTON:
    opt = nlopt_create(NLOPT_LD_TNEWTON, nParam);
    printf("Using NLOPT_LD_TNEWTON\n");
    break;
  case OptEnum::LN_PRAXIS:
    opt = nlopt_create(NLOPT_LN_PRAXIS, nParam);
    printf("Using NLOPT_LN_PRAXIS\n");
    break;
  case OptEnum::LN_NEWUOA:
    opt = nlopt_create(NLOPT_LN_NEWUOA, nParam);
    printf("Using NLOPT_LN_NEWUOA\n");
    break;
  case OptEnum::LN_COBYLA:
    opt = nlopt_create(NLOPT_LN_COBYLA, nParam);
    printf("Using NLOPT_LN_COBYLA\n");
    break;
  case OptEnum::LN_AUGLAG:
    opt = nlopt_create(NLOPT_LN_AUGLAG, nParam);
    printf("Using NLOPT_LN_AUGLAG\n");
    break;
  default:
    opt = nlopt_create(NLOPT_LD_LBFGS, nParam);
    printf("Unknown Opt Method ... defaulting to NLOPT_LD_LBFGS\n");
    break;
  }
    

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
  
  for(int i=0; i < nParam; i++) x[i] = oFuncVec->param[i];

  fini< DATA_TYPE, struct generatedFcnInterest<DATA_TYPE> >(paramfile.c_str(), oFuncVec);
  nlopt_destroy(opt);

  return 0;
}
