#ifndef UTIL_HPP
#define UTIL_HPP

#ifndef FCN_ATTRIBUTES
#define FCN_ATTRIBUTES
#endif

#include <iostream>
#include <cstring>
#include <vector>
#include <omp.h>
using namespace std;

inline double getTime() { return(omp_get_wtime());}

template< typename REAL_T, typename myFcnInterest >
struct ObjFuncVec {
  int nParam;
  REAL_T *param;
  vector< ObjFcn<REAL_T, myFcnInterest>* > vec;
  uint32_t nFunctionCalls;
  double timeObjFunction;
  double minTime, maxTime;
  double dataLoadTime;
  double err;

  ObjFuncVec()
  {
    myFcnInterest fi;
    nParam = fi.nParam();
    param = new REAL_T[nParam];

    nFunctionCalls = 0;
    minTime = maxTime = dataLoadTime = timeObjFunction = 0.;
  }
  ~ObjFuncVec()
  {
    delete [] param;
  }
};

extern "C" double nloptFunc(unsigned int n, const double *x,
			    double *grad, void* f_data)
{
  ObjFuncVec<float, generatedFcnInterest<float> >
    *oFuncVec = (ObjFuncVec<float, generatedFcnInterest<float> > *) f_data;
  
  assert(n == oFuncVec->nParam);

  double startTime = getTime();

#pragma SIMD
  for(int i=0; i < n; ++i)
    oFuncVec->param[i] = x[i];

  for(int i=0; i < oFuncVec->vec.size(); i++) {
    oFuncVec->vec[i]->offloadParam(oFuncVec->param);
  }
  
  // RUN on host if needed (possibly one task per thread)
  double err=0.;
  // nested parallelism as lower loop in func() is highly parallel 
  for(int i=0; i < oFuncVec->vec.size(); i++) {
    ObjFcn<float, generatedFcnInterest<float> > *oFunc = oFuncVec->vec[i];
    if(oFunc->devID >= 0) continue; 
    
    err += oFunc->func();
  }
  
  double timeDelta = getTime() - startTime;
  if(oFuncVec->nFunctionCalls == 1) { // ignore 0 function call
    oFuncVec->minTime = oFuncVec->maxTime = timeDelta;
  } 
  oFuncVec->timeObjFunction += timeDelta;
  oFuncVec->minTime = min(timeDelta, oFuncVec->minTime);
  oFuncVec->maxTime = max(timeDelta, oFuncVec->maxTime);
  oFuncVec->nFunctionCalls++;
  oFuncVec->err = err;

  return err;
}

int readParam(const char* filename, int nParam, float* param)
{
  FILE *fn=fopen(filename,"r");
  if(!fn) {
    return -1;
  }
  int parmInFile;
  int ret;
  
  ret=fread(&parmInFile,sizeof(uint32_t), 1, fn);
  if(ret != 1) return -1;

  if(parmInFile != nParam) {
    fprintf(stderr,"Number of parameters in file incorrect!\n");
    exit(1);
  }
  ret=fread(param,sizeof(float), nParam, fn);
  //if(ret != sizeof(float)) throw "parameter read failed";

  return 0;
}

void writeParam(const char *filename, int nParam, float *x)
{
  FILE *fn=fopen(filename,"w");
  if(!fn) {
    fprintf(stderr,"Cannot open %s\n",filename);
    exit(1);
  }

  int ret;

  ret=fwrite(&nParam,sizeof(uint32_t), 1, fn);
  if(ret != 1) throw "parameter write failed";
  ret=fwrite(x,sizeof(float), nParam, fn);
  if(ret != nParam) throw "parameter write failed";
  fclose(fn);
}

template< typename REAL_T, typename myFcnInterest >
ObjFuncVec<REAL_T, myFcnInterest >* init( const char* datafile,
					  const char * paramfile,
					  bool hostOnly)
{
  FILE *fn=stdin;

  cout << "*******************" << endl;

  ObjFuncVec<REAL_T, myFcnInterest >  *oFuncVec
    = new ObjFuncVec<REAL_T, myFcnInterest >;

  if(strcmp("-", datafile) != 0)
    fn=fopen(datafile,"r");
  
  if(!fn) {
    fprintf(stderr,"Cannot open %s\n",datafile);
    exit(1);
  }

  // read the header information
  double startTime=getTime();
  uint32_t nInput, nOutput;
  uint32_t nExamples;
  int ret;
  
  // read header
  ret=fread(&nInput,sizeof(uint32_t), 1, fn);
  assert(ret == 1);
  ret=fread(&nOutput,sizeof(uint32_t), 1, fn);
  assert(ret == 1);
  ret=fread(&nExamples,sizeof(uint32_t), 1, fn);
  assert(ret == 1);

  cout << "nInput " << nInput
       << " nOutput " << nOutput
       << " nExamples " << nExamples
       << " in datafile (" << datafile << ")"
       << endl;

  //read parameter file if it exists otherwise randomize the paramters
  for(int i=0; i < oFuncVec->nParam; i++) oFuncVec->param[i]
					    = 0.4*(rand()/(double)RAND_MAX);
  readParam(paramfile, oFuncVec->nParam, oFuncVec->param);

  vector<pair<int,int> > examplesPerDevice;

  // construct example vector
  examplesPerDevice.push_back(make_pair(-1,nExamples));

  // have examplesPerDevice constructed so read data into host-side oFunc
  for(int i=0; i < examplesPerDevice.size(); i++) {
    int myExamples = examplesPerDevice[i].second;
    assert(myExamples > 0);
    ObjFcn< REAL_T, myFcnInterest > *oFunc =
      new ObjFcn< REAL_T, myFcnInterest >(myExamples);
    
    // read the data
    for(int exIndex=0; exIndex < myExamples; exIndex++) {
      for(int i=0; i < nInput; i++) {
	ret=fread(& oFunc->InputExample(exIndex,i),1, sizeof(REAL_T), fn);
	//if(ret != sizeof(REAL_T)) throw "data read failed";
      }
      for(int i=0; i < nOutput; i++)  {
	ret=fread(& oFunc->KnownExample(exIndex,i),1, sizeof(REAL_T), fn);
	//if(ret != sizeof(REAL_T)) throw "data read failed";
      }
    }
    int dev = examplesPerDevice[i].first;
    
    if(examplesPerDevice[i].first < 0) {
      oFuncVec->vec.push_back( oFunc ); 
    } else { // unknown device
      cerr << "somehow got non-host id! " << examplesPerDevice[i].first;
      exit(1);
    }
    { //create and update example data on device, also create param on device as devID is set
      REAL_T *pInput = *oFunc->InputExample().getDataPtrAddr();
      REAL_T *pKnown = *oFunc->KnownExample().getDataPtrAddr();
      int sizeInput=myExamples*nInput;
      int sizeKnown=myExamples*nOutput;

      #pragma acc update device(pInput[0:sizeInput])
      #pragma acc update device(pKnown[0:sizeKnown])
    }
  }
  oFuncVec->dataLoadTime = getTime() - startTime;

  if(fn!=stdin)
    fclose(fn);
  
  assert(oFuncVec->vec.size() > 0);
  {
    myFcnInterest fi;
    cout << "Objective Function: " << oFuncVec->vec[0]->name() << endl
	 << "Function of Interest: " << fi.name()
	 << " with G() " << fi.gFcnName() << endl;
  }
  
  return(oFuncVec);
}

template< typename REAL_T, typename myFcnInterest >
void fini(const char * paramFilename,
	  ObjFuncVec<REAL_T, myFcnInterest >* oFuncVec)
{
  double nExamples = 0.;
  for(int i=0; i < oFuncVec->vec.size(); i++)
    nExamples += oFuncVec->vec[i]->InputExample().rows();
  
  uint32_t nFlops = oFuncVec->vec[0]->FcnInterest_nFlops();
  double averageTimeObjFunc =
    ((double) oFuncVec->timeObjFunction)/(oFuncVec->nFunctionCalls-1.);
  double totalFlops = ((double) nExamples) * ((double) nFlops);
  
  printf("RUNTIME Info (%g examples)\n",nExamples);
  printf("\tDataLoadtime %g seconds\n", oFuncVec->dataLoadTime);
  printf("\tAveObjTime %g, countObjFunc %d, totalObjTime %g\n",
	 averageTimeObjFunc, oFuncVec->nFunctionCalls,
	 oFuncVec->timeObjFunction);
#ifdef FLOP_ESTIMATE
  printf("\tEstimated Flops myFunc %d,average GFlop/s %g nFuncCalls %d\n",
	 nFlops, (totalFlops/averageTimeObjFunc/1.e9), 
	 oFuncVec->nFunctionCalls);
  printf("\tEstimated maximum GFlop/s %g, minimum GFLop/s %g\n",
	 (totalFlops/(oFuncVec->minTime)/1.e9),
	 (totalFlops/(oFuncVec->maxTime)/1.e9) );
#endif

  writeParam(paramFilename, oFuncVec->nParam, oFuncVec->param);

  delete oFuncVec;
}

#endif

