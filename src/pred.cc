#include <iostream>
using namespace std;

#include "DataType.hpp"
#include "FcnOfInterest.hpp"
#include "ObjFcn.hpp"
#include <algorithm>
#include <nlopt.h>
#include "util.hpp"
#include <getopt.h>

extern double getTime();

void PrintHelp(char *PgmName)
{
  cout <<
    "************************** " << PgmName  << " ********************************\n"
    "--data <string>:\tname of the data file. REQUIRED\n"
    "--param <string>:\tname of the param file. REQUIRED\n"
    "--help:\tShow help\n";
  exit(1);
}

// global commandline values;
string datafile, paramfile;

int ProcessArgs(int argc, char *argv[])
{
  if(argc == 1) {
    PrintHelp(argv[0]);
    return -1;
  }
  const char* const short_opts = "d:p:h";
  const option long_opts[] = {
    {"data", required_argument, nullptr, 'd'},
    {"param", required_argument, nullptr, 'p'},
    {"help", no_argument, nullptr, 'h'},
    {nullptr, no_argument, nullptr, 0}
  };
  while(true) {
    const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);
    if(-1 == opt)
      break;

    switch(opt) {
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
  // load the data and parameters
  ObjFuncVec<DATA_TYPE, struct generatedFcnInterest<DATA_TYPE> > *oFuncVec =
    init< DATA_TYPE, struct generatedFcnInterest<DATA_TYPE> > (datafile.c_str(), paramfile.c_str(),true);

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
