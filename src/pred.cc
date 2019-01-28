#include <iostream>
using namespace std;

#include "FcnOfInterest.hpp"
#include <algorithm>
#include <getopt.h>

#ifndef PREDFCN
#define PREDFCN generic_fcn
#endif

class PredFcn {
private:
  vector<float> param;
  PredFcn();
public:
  generatedFcnInterest<float> fi;
  PredFcn(const char * paramFile) { loadParam(paramFile); }
  PredFcn(const string &paramFile) { loadParam(paramFile.c_str()); }
  void loadParam(const char * filename)
  {
    FILE *fn=fopen(filename,"r");
    if(!fn) throw runtime_error("Cannot open file");

    int parmInFile;
    int ret;
    
    ret=fread(&parmInFile,sizeof(uint32_t), 1, fn);
    if(ret != 1) throw runtime_error("header read failure in parameter file");
    
    if(parmInFile != fi.nParam()) {
      if(ret != fi.nParam()) throw runtime_error("Incorrect number of parameters in file");
    }
    param.reserve(fi.nParam());
    ret=fread(&param[0],sizeof(float), fi.nParam(), fn);
    if(ret != fi.nParam()) throw runtime_error("parameter read failed");
  }

  void predict(const float *input, float * output)
  {
    fi.PREDFCN<true,float>(&param[0],input,output);
  }
};

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
  cerr << "predict data in: " << datafile << endl;
  if(paramfile.empty()) {
    cout << "Fatal: must specify a parameter file " << endl;
    return -1;
  }

  try {
    PredFcn ann(paramfile);
    
    // Now read the input stream/file.
    int nInput, nOutput;
    int ret;
    FILE *fn=stdin;
    
    if(datafile != "-")
      fn=fopen(datafile.c_str(),"r");
    
    if(!fn) {
      cerr << "Cannot open %s " << datafile << endl;
      exit(1);
    }
    
    ret=fread(&nInput,sizeof(int), 1, fn);
    if(ret != 1) throw runtime_error("datafile read failed");
    if(nInput != ann.fi.nInput())
      throw runtime_error("Incorrect number of inputs");
    
    // we accept pure predict streams with nOutput of zero in the datafile header
    ret=fread(&nOutput,sizeof(int), 1, fn);
    if(ret != 1) throw runtime_error("datafile read failed");
    if( (nOutput > 0) && (nOutput != ann.fi.nOutput()) )
      throw runtime_error("Incorrect number of Outputs");

    // Ignore the nExamples header.
    int tmp;
    ret=fread(&tmp,sizeof(int), 1, fn);
    if(ret != 1) throw runtime_error("datafile read failed");
    
    vector<float> I, Known,O;
    I.reserve(nInput);
    O.reserve(nInput);
    Known.reserve(nOutput);

    if(nOutput == 0) O.reserve(nInput);
    else O.reserve(nOutput);

    for(;;) {
      ret=fread(&I[0],sizeof(float), nInput, fn);
      if(ret != nInput) break;
      if(nOutput > 0) {
	ret=fread(&Known[0],sizeof(float), nOutput, fn);
	if(ret != nOutput) break;
      }
      ann.predict(&I[0],&O[0]);

      // output in CSV format
      if(nOutput == 0) { // autoencoder special case
	cout << "pred";
	for(int j=0; j < nInput; j++) cout << ", " << O[j];
	cout << ", known"; 
	for(int j=0; j < nInput; j++) cout << ", " << I[j];
	cout << endl;
      } else {
	cout << "pred"; 
	for(int j=0; j < nOutput; j++) cout << ", " << O[j];
	cout << ", known"; 
	for(int j=0; j < nOutput; j++) cout << ", " << Known[j];
	cout << endl;
      }
    }
  } catch(const runtime_error &e) {
    cout << "Exception found " << e.what() << endl;
  }
  
  
  
  return 0;
}


