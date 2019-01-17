#ifndef UTIL_HPP
#define UTIL_HPP

//Specify the min compute arch
#define MIN_ARCH_MAJOR 3
#define MIN_ARCH_MINOR 3

#include <iostream>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <omp.h>
using namespace std;

#ifdef USE_MPI

#warning "MPI not tested on real hardware"
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


__global__ void launchObjKernel( ObjFcn<float,
				 generatedFcnInterest<float> >* obj)
{
  obj->cuda_func();
}

template< typename REAL_T, typename myFcnInterest >
struct ObjFuncVec {
private:
  string paramFilename;
  double prevTime, last_err;
  int checkpointInterval;

public:
  int nParam;
  REAL_T *param;
  vector< ObjFcn<REAL_T, myFcnInterest>* > vec;
  uint32_t nFunctionCalls;
  double timeObjFunction;
  double minTime, maxTime;
  double dataLoadTime;
  double err;
  uint32_t totalExamples;
#ifdef USE_GRAD
  double timeGrad;
  uint32_t nGradCalls;
  vector< ObjFcn<REAL_T, myFcnInterest>* > grad_data;
#endif

  ObjFuncVec()
  {
    myFcnInterest fi;
    nParam = fi.nParam();
    cudaHostAlloc(&param, nParam*sizeof(REAL_T), cudaHostAllocDefault); 

    nFunctionCalls = 0;
    minTime = maxTime = dataLoadTime = timeObjFunction = 0.;
    checkpointInterval=0; // default is no checkpoint
    prevTime = 0.;
    last_err=0.;
    totalExamples=0;
#ifdef USE_GRAD
    timeGrad=0.;
    nGradCalls=0;
#endif
  }
  ~ObjFuncVec()
  {
    cudaFreeHost(param);
  }
  void setParamFilename(const char *fname)
  {
    paramFilename = fname;
    paramFilename.erase(paramFilename.find_last_of(".", string::npos));
  }
  // checkpoint every interval of runtime
  void setCheckPointInterval(int interval_seconds)
  {
    if(interval_seconds < 0) throw "invalid interval_seconds specified";
#ifdef USE_MPI
    if(getMPI_rank() == 0) { // master
      checkpointInterval = interval_seconds;
    }
#else
    checkpointInterval = interval_seconds;
#endif
  }

  string getParamCheckPointName()
  {
    time_t rawtime;
    struct tm timeinfo;
    
    time (&rawtime);
    gmtime_r (&rawtime, &timeinfo);
    
    char buffer [80];
    strftime(buffer, 80, ".utc-%Y_%m_%d_%H_%M_%S", &timeinfo);
    char err_s[80];
    sprintf(err_s, ".err-%.10lf",err);
    return(paramFilename+buffer+err_s+".dat");
  }
  void writeCheckpoint(double curTime)
  {
    if(last_err == 0.) last_err = err;

    if(checkpointInterval && nFunctionCalls && (err <= last_err) 
       && (((curTime - prevTime) >= checkpointInterval) || curTime==0.)) {
      extern void writeParam(const char *, int , float *);
      writeParam(getParamCheckPointName().c_str(),nParam, param);
      writeParam((paramFilename+".dat").c_str(),nParam, param);
      prevTime = getTime();
      last_err = err;
    }
  }
  void cuda_PartitionLoadData(uint32_t nExamples, bool hostOnly, uint32_t nInput, uint32_t nOutput, FILE *fn)
  {
    int ret;
    vector<pair<int,int> > examplesPerDevice;
    
    // construct example vector
    if(hostOnly) {
      //Note: set from command-line to size for hybrid CPU/CPU compute
      examplesPerDevice.push_back(make_pair(-1,nExamples));
    }
    
    vector<cudaDeviceProp> propsVec;
    if(!hostOnly) {
      int nDevices;
      cudaGetDeviceCount(&nDevices);
      int totalCudaExamples=nExamples;
      // subtract off all host examples
      for(int i=0; i < examplesPerDevice.size(); i++)
	totalCudaExamples -= examplesPerDevice[i].second;
      assert(totalCudaExamples > 0);
      
      // Get device properties
      for(int dev=0; dev < nDevices; dev++) {
	cudaDeviceProp props;
#ifdef USE_CUDA_DEVICE
#warning "USING a single CUDA device"
	if(dev != USE_CUDA_DEVICE) continue;
#endif
	cudaSetDevice(dev);
	assert(cudaError() == cudaSuccess);
	cudaGetDeviceProperties(&props, dev);
	assert(cudaError() == cudaSuccess);
	propsVec.push_back(props);
      }
      
      // find number of active blocks to decide how many examples per device
      int totalActiveBlk=0;
      for(int i=0; i < propsVec.size(); i++) {
	int nBlocks = propsVec[i].maxThreadsPerBlock/WARP_SIZE
	  * propsVec[i].multiProcessorCount;
	totalActiveBlk += nBlocks;
      }
      
      int examplesPerBlock = totalCudaExamples/totalActiveBlk; 
      for(int i=propsVec.size()-1; i >= 0; i--) {
	// Ignore devices less than 3.5
	if(propsVec[i].major < MIN_ARCH_MAJOR) continue;
	if(propsVec[i].major == MIN_ARCH_MAJOR
	   && propsVec[i].minor < MIN_ARCH_MINOR) continue;
	
	int nBlocks = propsVec[i].maxThreadsPerBlock/WARP_SIZE
	  * propsVec[i].multiProcessorCount;
	
	int n = nBlocks * examplesPerBlock;
	
	totalCudaExamples -= n;
	assert(totalCudaExamples >= 0);
	if(i==0) {
	  examplesPerDevice.push_back(make_pair(i,n+totalCudaExamples));
	} else if(n>0) {
	  examplesPerDevice.push_back(make_pair(i,n));
	}
      }
    }
    
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
	  if(ret != sizeof(REAL_T)) throw "data read failed";
	}
	for(int i=0; i < nOutput; i++)  {
	  ret=fread(& oFunc->KnownExample(exIndex,i),1, sizeof(REAL_T), fn);
	  if(ret != sizeof(REAL_T)) throw "data read failed";
	}
      }
      int dev = examplesPerDevice[i].first;
      
      if(examplesPerDevice[i].first < 0) {
	this->vec.push_back( oFunc ); 
      } else { // CUDA device
	oFunc->warpSize = propsVec[dev].warpSize;
	oFunc->multiProcessorCount = propsVec[dev].multiProcessorCount;
	oFunc->maxThreadsPerBlock = propsVec[dev].maxThreadsPerBlock;
	
	this->vec.push_back(new ObjFcn< REAL_T, myFcnInterest >(*oFunc, dev));
	
	printf("\tDevice %d: \"%s\" with Compute %d.%d capability\n",
	       dev, propsVec[dev].name,
	       propsVec[dev].major, propsVec[dev].minor);
	printf("\tWarpSize %d, nSMX %d,maxThreadsPerBlock %d\n",
	       oFunc->warpSize, oFunc->multiProcessorCount, 
	       oFunc->maxThreadsPerBlock);
	
#ifdef USE_GRAD
	// if using a gradient, keep the data
	this->grad_data.push_back(oFunc);
#else
	delete oFunc; // discard the host-side data to save space
#endif
      }
    }
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

#pragma SIMD
  for(int i=0; i < n; ++i)
    oFuncVec->param[i] = x[i];

  for(int i=0; i < oFuncVec->vec.size(); i++) {
    oFuncVec->vec[i]->offloadParam(oFuncVec->param);
  }
  
  double err=0.;
  bool runOnHostToo=false;
  for(int i=0; i < oFuncVec->vec.size(); i++) {
    ObjFcn<float, generatedFcnInterest<float> > *oFunc = oFuncVec->vec[i];
    if(oFunc->devID < 0) {
      runOnHostToo=true;
      continue; 
    }

    assert(oFunc->warpSize == 32);
    
    int nActiveBlks = oFunc->maxThreadsPerBlock/oFunc->warpSize;
    int nBlocks = oFunc->multiProcessorCount* nActiveBlks;

    cudaSetDevice(oFunc->devID);
    launchObjKernel<<<nBlocks, oFunc->warpSize>>>(oFunc->d_oFunc);

    if(cudaError() != cudaSuccess) throw "func failed";
  }

  // RUN on host if needed (possibly one task per thread)
  if(runOnHostToo) {
#pragma omp parallel for reduction(+:err) 
    for(int i=0; i < oFuncVec->vec.size(); i++) {
      ObjFcn<float, generatedFcnInterest<float> > *oFunc = oFuncVec->vec[i];
      if(oFunc->devID >= 0) continue; 
      
      err += oFunc->func();
    }
  }
  
  // zero the partial error on the device
  for(int i=0; i < oFuncVec->vec.size(); i++) {
    ObjFcn<float, generatedFcnInterest<float> > *oFunc = oFuncVec->vec[i];
    if(oFunc->devID < 0) continue; 

    cudaSetDevice(oFunc->devID);
    cudaMemcpy(&oFunc->myErr, &oFunc->d_oFunc->myErr,
	       sizeof(double),cudaMemcpyDeviceToHost);
    assert(cudaError() == cudaSuccess);
    err += oFunc->myErr;
  }
  
#ifdef USE_MPI
  double partialError = err;
  double totalError=0.;
  if(getMPI_rank() == 0) { // master
    MPI_Reduce(&partialError, &totalError, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // get the totalError
    err = totalError;
  }
#endif
  
  double curTime = getTime();
  double timeDelta = getTime() - startTime;
  if(oFuncVec->nFunctionCalls == 1) { // ignore 0 function call
    oFuncVec->minTime = oFuncVec->maxTime = timeDelta;
  } 
  oFuncVec->timeObjFunction += timeDelta;
  oFuncVec->minTime = min(timeDelta, oFuncVec->minTime);
  oFuncVec->maxTime = max(timeDelta, oFuncVec->maxTime);
  oFuncVec->nFunctionCalls++;
  oFuncVec->err = err;

  oFuncVec->writeCheckpoint(curTime);

  if(grad != NULL) {
#ifdef USE_GRAD
    double startTime = getTime();
    oFuncVec->nGradCalls++;
    // TODO: This does not use the host-side data for the gradient
    for(int i=0; i < oFuncVec->nParam; i++) grad[i]=0;
    
    for(int i=0; i < oFuncVec->grad_data.size(); i++) {
      ObjFcn<float, generatedFcnInterest<float> > *oFunc = oFuncVec->grad_data[i];
      oFunc->gen_grad(grad,x);
    }
    oFuncVec->timeGrad += getTime() - startTime;
#else
    printf("GRAD not used\n");
    assert(false);
#endif
  }
  
#warning "test that totalExamples using multiple GPUs and/or MPI is correct" << endl; 
  //cerr << "TotalCudaExamples " << oFuncVec->totalExamples << endl;
  return err/oFuncVec->totalExamples;
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
  
  ObjFuncVec<REAL_T, myFcnInterest >  *oFuncVec
    = new ObjFuncVec<REAL_T, myFcnInterest >;

  if(strcmp("-", datafile) != 0)
    fn=fopen(datafile,"r");
  
  if(!fn) {
    fprintf(stderr,"Cannot open %s\n",datafile);
    exit(1);
  }

  oFuncVec->setParamFilename(paramfile);

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
  oFuncVec->totalExamples=nExamples;

#ifdef DEBUG_DATA_PARTITIONING
  {
    FILE *fn=fopen("data.hdr","w");
    ret=fwrite(&nInput,sizeof(uint32_t), 1, fn);
    assert(ret == 1);
    ret=fwrite(&nOutput,sizeof(uint32_t), 1, fn);
    assert(ret == 1);
    ret=fwrite(&nExamples,sizeof(uint32_t), 1, fn);
    fclose(fn);
  }
#endif

#ifdef USE_MPI
  int nTasks = getMPI_tasks();
  int rank = getMPI_rank();

  if(rank==0) { // master
    cout << "nInput " << nInput
	 << " nOutput " << nOutput
	 << " nExamples " << nExamples
	 << " in datafile (" << datafile << ")"
	 << endl;
    cout << "*******************" << endl;
#pragma omp parallel
    if(omp_get_thread_num() == 0)
      cout << "Using MPI" << " numTasks " <<  nTasks << " OMP_NUM_THREADS " << omp_get_num_threads() << endl;
  }
  
  {
    // Set rankExamples vector to the number of examples per rank.
    vector<int> rankExample(nTasks);
    for(int i=0; i < nTasks; i++) rankExample[i] = nExamples/nTasks;

    for(int i=0; i < nExamples % nTasks; i++) rankExample[i]++;

    uint32_t offset=0;
    for(int i=0; i < rank; i++) offset += rankExample[i];
    // test if seekable
    // seek to location
    if(fseek(fn, (nInput+nOutput)*offset*sizeof(REAL_T), SEEK_CUR) != 0) {
      cerr << "Cannot seek to location" << endl;
      exit(1);
    }
    
    if(rank==0) { // master
      cout << "\tExamples MPI rank 0 " << rankExample[0] 
	   << "\tExamples MPI rank " << (nTasks-1) << " " << rankExample[nTasks-1] << endl; 
    }

    offset=0;
    for(int i=0; i < nTasks; i++) offset += rankExample[i];
    assert(offset == nExamples);

    // set nExamples to the right one for the current rank
    nExamples = rankExample[rank];
  }
#else
#pragma omp parallel
  if(omp_get_thread_num() == 0)
    cout << "OMP_NUM_THREADS " << omp_get_num_threads() << endl;
  cout << "nInput " << nInput
       << " nOutput " << nOutput
       << " nExamples " << nExamples
       << " in datafile (" << datafile << ")"
       << endl;
  cout << "*******************" << endl;
#endif

  oFuncVec->totalExamples = nExamples;
  oFuncVec->cuda_PartitionLoadData(nExamples, hostOnly, nInput, nOutput,fn);

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
#ifdef USE_GRAD
  if(oFuncVec->nGradCalls > 0) {
    printf("\tAveGradTime %g, nGradCalls %d, totalGradTime %g\n",
	   (oFuncVec->nGradCalls==0)?0.:oFuncVec->timeGrad/oFuncVec->nGradCalls,
	   oFuncVec->nGradCalls,
	   oFuncVec->timeGrad );
    printf("\tnFuncCalls/nGradCalls %3.2f\n",
	   ((double)oFuncVec->nFunctionCalls)/((double) oFuncVec->nGradCalls));
  }
#endif
  
  writeParam(paramFilename, oFuncVec->nParam, oFuncVec->param);
  oFuncVec->writeCheckpoint(0.);
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


