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

#ifdef USE_MPI

#include "mpi.h"

inline int getMPI_rank() {
  int mpiRank;;
  MPI_Comm_rank(MPI_COMM_WORLD,&mpiRank);
  return mpiRank;;
}

inline int getMPI_tasks() {
  int numTasks;
  MPI_Comm_size(MPI_COMM_WORLD,&numTasks);
  return numTasks;
}
#endif

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

#ifdef USE_MPI
  if(getMPI_rank() == 0) { // master
    int op=1;
    MPI_Bcast(&op, 1, MPI_INT, 0, MPI_COMM_WORLD); // Send the master op code
    MPI_Bcast((void*) x, N_PARAM, MPI_DOUBLE, 0, MPI_COMM_WORLD); // Send the parameters
  }
#endif

#pragma simd
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

#ifdef USE_MPI
  double partialError = err;
  double totalError=0.;
  if(getMPI_rank() == 0) { // master
    MPI_Reduce(&partialError, &totalError, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // get the totalError
    err = totalError;
  }
#endif
  
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

#ifdef USE_MPI
void startClient(double *xFromMPI, void *my_func_data)
{
  int op;
  double partialError,sum;

  
  for(;;) { // loop until the master says I am done
    MPI_Bcast(&op, 1, MPI_INT, 0, MPI_COMM_WORLD); // receive the op code
    if(op==0) { // we are done, normal exit
      break;
    }
    MPI_Bcast(xFromMPI, N_PARAM, MPI_DOUBLE, 0, MPI_COMM_WORLD); // receive the parameters
    partialError = nloptFunc(N_PARAM,  xFromMPI, NULL, my_func_data);
    MPI_Reduce(&partialError, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  MPI_Finalize();
}
#endif


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

// mpiInit here 
// send filename to each rank and number of ranks
// each rank opens file and reads header, then seeks to offset = nExamples/nRanks * exampleSize

template< typename REAL_T, typename myFcnInterest >
ObjFuncVec<REAL_T, myFcnInterest >* init( const char* datafile,
					  const char * paramfile,
					  bool hostOnly)
{
  FILE *fn=stdin;

#ifdef USE_MPI
  {
    int ret = MPI_Init(NULL,NULL);
    
    if (ret != MPI_SUCCESS) {
      printf ("Error in MPI_Init()!\n");
      MPI_Abort(MPI_COMM_WORLD, ret);
    }
  }
#endif
  
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
  int rank=0;
  
  // read header
  ret=fread(&nInput,sizeof(uint32_t), 1, fn);
  assert(ret == 1);
  ret=fread(&nOutput,sizeof(uint32_t), 1, fn);
  assert(ret == 1);
  ret=fread(&nExamples,sizeof(uint32_t), 1, fn);
  assert(ret == 1);

  vector<pair<int,int> > examplesPerDevice;

#ifdef USE_MPI
  int nTasks = getMPI_tasks();
  rank = getMPI_rank();

  if(rank==0) // master
    cout << "Using MPI" << " numTasks " <<  nTasks << " OMP_NUM_THREADS " << omp_get_num_threads() << endl;
  
  int rankExample = nExamples/getMPI_tasks();
  int lastrankExample = (nTasks*rankExample > nExamples)?nExamples - (nTasks-1)*rankExample:nExamples;
  if(rank+1 == nTasks) { // last rank so have to adjust to nExamples
    rankExample = lastrankExample;
  }
  assert(rankExample*(nTasks-1) == nExamples);
  // test if seekable
  if(fseek(fn, (nInput+nOutput)*rankExample, SEEK_SET) != 0) {
    cerr << "Cannot seek to location" << endl;
    exit(1);
  }
  // seek to location
  nExamples = rankExample;
  if(rank==0) // master
    cout << "\t examples per MPI rank " << rankExample << endl; 
#endif

  if(rank==0) // master
    cout << "nInput " << nInput
	 << " nOutput " << nOutput
	 << " nExamples " << nExamples
	 << " in datafile (" << datafile << ")"
	 << endl;
  
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
  }
  oFuncVec->dataLoadTime = getTime() - startTime;

  if(fn!=stdin)
    fclose(fn);
  
#ifdef USE_MPI
  if(getMPI_rank() > 0) {
    vector<double> x(N_PARAM);
    startClient(&x[0], (void *) oFuncVec);
    exit(0); // normal client exit
  }
#endif

  //read parameter file if it exists otherwise randomize the paramters
  for(int i=0; i < oFuncVec->nParam; i++) oFuncVec->param[i]
					    = 0.4*(rand()/(double)RAND_MAX);
  readParam(paramfile, oFuncVec->nParam, oFuncVec->param);

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
#ifdef USE_MPI
  {
    int op=0;
    MPI_Bcast(&op, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
  }
#endif

  delete oFuncVec;
}

#endif

