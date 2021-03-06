#ifndef UTIL_HPP
#define UTIL_HPP

#include <iostream>
#include <cstring>
#include <vector>
#include <omp.h>
using namespace std;

inline double getTime() { return(omp_get_wtime());}

__global__ void launchParamConvert( ObjFcn<DATA_TYPE,
				    generatedFcnInterest<DATA_TYPE> >* obj)
{
  obj->paramConvert();
}

__global__ void launchObjKernel( ObjFcn<DATA_TYPE,
				 generatedFcnInterest<DATA_TYPE> >* obj)
{
  obj->cuda_func();
}

__global__ void launchDataConvert(DATA_TYPE *data, float *f_data, uint32_t n)
{
  DATA_TYPE::convertFromFloat(data, f_data, n);
}

template< typename REAL_T, typename myFcnInterest >
struct ObjFuncVec {
  int nParam;
  float *param;
  float *d_param;
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
    cudaHostAlloc(&param, nParam*sizeof(float), cudaHostAllocDefault); 

    nFunctionCalls = 0;
    minTime = maxTime = dataLoadTime = timeObjFunction = 0.;
  }
  ~ObjFuncVec()
  {
    cudaFreeHost(param);
  }
  uint32_t totalExamples()
  {
    uint32_t nExamples = 0;
    for(int i=0; i < vec.size(); i++) nExamples += vec[i]->InputExample().rows();
    return DATA_TYPE_VLEN * nExamples;
  }
};
  
extern "C" double nloptFunc(unsigned int n, const double *x,
			    double *grad, void* f_data)
{
  ObjFuncVec<DATA_TYPE, generatedFcnInterest<DATA_TYPE> >
    *oFuncVec = (ObjFuncVec<DATA_TYPE, generatedFcnInterest<DATA_TYPE> > *) f_data;
  
  assert(n == oFuncVec->nParam);

  double startTime = getTime();

#pragma SIMD
  for(int i=0; i < n; ++i)
    oFuncVec->param[i] = x[i];

  for(int i=0; i < oFuncVec->vec.size(); i++) {
    oFuncVec->vec[i]->offloadParam(oFuncVec->param);
  }
  
  double err=0.;
  bool runOnHostToo=false;
  for(int i=0; i < oFuncVec->vec.size(); i++) {
    ObjFcn<DATA_TYPE, generatedFcnInterest<DATA_TYPE> > *oFunc = oFuncVec->vec[i];
    if(oFunc->devID < 0) {
      runOnHostToo=true;
      continue; 
    }

    assert(oFunc->warpSize == 32);
    
    int nActiveBlks = oFunc->maxThreadsPerBlock/oFunc->warpSize;
    int nBlocks = oFunc->multiProcessorCount* nActiveBlks;

    cudaSetDevice(oFunc->devID);
    launchObjKernel<<<nBlocks, oFunc->warpSize>>>(oFunc->d_oFunc);

    if(cudaGetLastError() != cudaSuccess) throw "func failed";
  }

  // RUN on host if needed (possibly one task per thread)
  if(runOnHostToo) {
#pragma omp parallel for reduction(+:err) 
    for(int i=0; i < oFuncVec->vec.size(); i++) {
      ObjFcn<DATA_TYPE, generatedFcnInterest<DATA_TYPE> > *oFunc = oFuncVec->vec[i];
      if(oFunc->devID >= 0) continue; 
      
      err += oFunc->func();
    }
  }
  
  // zero the partial error on the device
  for(int i=0; i < oFuncVec->vec.size(); i++) {
    ObjFcn<DATA_TYPE, generatedFcnInterest<DATA_TYPE> > *oFunc = oFuncVec->vec[i];
    if(oFunc->devID < 0) continue; 

    cudaSetDevice(oFunc->devID);
    cudaMemcpy(&oFunc->myErr, &oFunc->d_oFunc->myErr,
	       sizeof(double),cudaMemcpyDeviceToHost);
    assert(cudaGetLastError() == cudaSuccess);
    err += oFunc->myErr;
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
  //if(ret != nParam) throw "parameter read failed";

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
  size_t ret;
  
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
#ifdef LAYOUT_SOA
  cout << "Using SOA layout" << endl;
#endif
  cout << "Using vector length " << DATA_TYPE_VLEN << endl;

  //read parameter file if it exists otherwise randomize the paramters
  for(int i=0; i < oFuncVec->nParam; i++) {
    oFuncVec->param[i] = 0.4*(rand()/(double)RAND_MAX);
  }

  vector<float> tmp;
  tmp.resize(oFuncVec->nParam);
  if(readParam(paramfile, oFuncVec->nParam, &tmp[0]) >= 0) {
    for(int i=0; i < oFuncVec->nParam; i++)
      oFuncVec->param[i] = tmp[i];
  }

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
      assert(cudaGetLastError() == cudaSuccess);
      cudaGetDeviceProperties(&props, dev);
      assert(cudaGetLastError() == cudaSuccess);
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
  if(examplesPerDevice.size() == 0) {
    std::cerr << "No CUDA " << MIN_ARCH_MAJOR << "." << MIN_ARCH_MINOR
	      << " compute devices found!" << std::endl;
    exit(1);
  }
    
  // have examplesPerDevice constructed so read data into host-side oFunc
  for(int i=0; i < examplesPerDevice.size(); i++) {
    int myExamples = examplesPerDevice[i].second;
    assert(myExamples > 0);
    ObjFcn< REAL_T, myFcnInterest > *oFunc =
      new ObjFcn< REAL_T, myFcnInterest >(myExamples);

    float *h_f_buf, *d_f_buf;
    REAL_T *h_type_buf, *d_type_buf;

    // **********
    // no host side conversions to/from fp16 so use device
    // **********
    
    if(examplesPerDevice[i].first < 0) {
      cerr << "Host vector packin not implemented" << endl;
      exit(1);
    }
    cudaSetDevice(examplesPerDevice[i].first);
    // allocate data on host and device
    uint32_t totalSize = myExamples*(nInput+nOutput);
    uint32_t floatBytes = totalSize * sizeof(float);
    uint32_t typeBytes = totalSize/DATA_TYPE_VLEN * sizeof(REAL_T);

    cudaHostAlloc(&h_f_buf, floatBytes, cudaHostAllocDefault); 
    cudaMalloc(&d_f_buf, floatBytes); 
    cudaHostAlloc(&h_type_buf, typeBytes, cudaHostAllocDefault); 
    cudaMalloc(&d_type_buf, typeBytes); 
    if(cudaGetLastError() != cudaSuccess) throw "cudaMalloc failed";

    // repack i0,k0,i1,k1, ... into vector format where adjacent
    // examples sit next to each other one per vector lane.
    {
      int packedIndex=0;
      vector<float> buf;
      buf.resize(DATA_TYPE_VLEN*(nInput+nOutput));
      for(int exIndex=0; exIndex < myExamples/DATA_TYPE_VLEN; exIndex++) {
	for(int k=0; k < DATA_TYPE_VLEN; k++) {
	  int index=k;
	  for(int i=0; i< nInput; i++) {
	    ret=fread(&buf[index],sizeof(float), 1, fn);
	    if(ret != 1) throw "data read failed";
	    index += DATA_TYPE_VLEN;
	  }
	  for(int i=0; i< nOutput; i++) {
	    ret=fread(&buf[index],sizeof(float), 1, fn);
	    if(ret != 1) throw "data read failed";
	    index += DATA_TYPE_VLEN;
	  }
	}
	for(int i=0; i < buf.size(); i++) {
	  assert(packedIndex < totalSize);
	  h_f_buf[packedIndex++] = buf[i];
	}
      }
    }

    // copy float data to device
    cudaMemcpy(d_f_buf, h_f_buf, floatBytes, cudaMemcpyHostToDevice);
    if(cudaGetLastError() != cudaSuccess) throw "memcpy failed";

    // convert on device (assume no host based conversion because of FP16)
    int nBlocks = myExamples/32 + 1; 
    launchDataConvert <<<nBlocks, 32>>>(d_type_buf, d_f_buf, totalSize);
    cudaDeviceSynchronize();
    if(cudaGetLastError() != cudaSuccess) throw "launchDataConvert failed!";

    // copy converted data back to host
    cudaMemcpy(h_type_buf, d_type_buf, typeBytes, cudaMemcpyDeviceToHost);
    if(cudaGetLastError() != cudaSuccess) throw "memcpy failed";
      
    myExamples = examplesPerDevice[i].second /= DATA_TYPE_VLEN;
    assert(myExamples % DATA_TYPE_VLEN == 0);

    // assign data according to indexing structure
    uint32_t h_buf_index=0;
    for(int exIndex=0; exIndex < myExamples; exIndex++) {
      for(int i=0; i < nInput; i++)
	oFunc->InputExample(exIndex,i) = h_type_buf[h_buf_index++];
      for(int i=0; i < nOutput; i++)
	oFunc->KnownExample(exIndex,i) = h_type_buf[h_buf_index++];
    }
    // free
    cudaFreeHost(h_f_buf); cudaFreeHost(h_type_buf);
    cudaFree(d_f_buf); cudaFree(d_type_buf);

    int dev = examplesPerDevice[i].first;

    if(examplesPerDevice[i].first < 0) {
      oFuncVec->vec.push_back( oFunc ); 
    } else { // CUDA device
      oFunc->warpSize = propsVec[dev].warpSize;
      oFunc->multiProcessorCount = propsVec[dev].multiProcessorCount;
      oFunc->maxThreadsPerBlock = propsVec[dev].maxThreadsPerBlock;

      oFuncVec->vec.push_back(new ObjFcn< REAL_T, myFcnInterest >(*oFunc, dev));

      printf("\tDevice %d: \"%s\" with Compute %d.%d capability\n",
	     dev, propsVec[dev].name,
	     propsVec[dev].major, propsVec[dev].minor);
      printf("\tWarpSize %d, nSMX %d,maxThreadsPerBlock %d\n",
	     oFunc->warpSize, oFunc->multiProcessorCount, 
	     oFunc->maxThreadsPerBlock);
      
      delete oFunc;
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
  double nExamples = oFuncVec->totalExamples();
  
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


