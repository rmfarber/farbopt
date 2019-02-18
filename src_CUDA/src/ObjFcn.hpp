#ifndef OBJFCN_HPP
#define OBJFCN_HPP

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include "Matrix.hpp"

#ifdef USE_GRAD
#include <adolc/adolc.h>
#ifdef _OPENMP
#include <omp.h>
#include <adolc/adolc_openmp.h>
#endif
#endif

#define WARP_SIZE 32

#if __CUDA_ARCH__ < 600
__device__ inline void cuda_atomicAdd (double *address, double value)
{
  unsigned long long oldval, newval, readback; 
  
  oldval = __double_as_longlong(*address);
  newval = __double_as_longlong(__longlong_as_double(oldval) + value);
  while ((readback=atomicCAS((unsigned long long *)address, oldval, newval)) != oldval)
    {
      oldval = readback;
      newval = __double_as_longlong(__longlong_as_double(oldval) + value);
    }
}
#endif

template< typename REAL_T, typename myFcnInterest >
class ObjFcn {
private:
  myFcnInterest fi;
  Matrix<REAL_T> Input, Known; 
  REAL_T *param;
  ObjFcn() { }
  bool have_tape;
  
#ifdef USE_GRAD
  void createAdolcTape(int tag, bool warn)
  {
    uint32_t nExamples = Input.rows();
    for(int i=0; i < fi.nParam(); i++) param[i]=0;

    trace_on(tag,1);
    adouble *ad_param = new adouble[fi.nParam()];
    for (int i=0; i< fi.nParam(); i++) ad_param[i] <<= param[i];

    adouble *ad_I = new adouble[fi.nInput()];
    adouble *ad_K = new adouble[fi.nOutput()];
    for(int i=0; i< fi.nInput(); i++) ad_I[i] = mkparam(Input(0,i));
    for(int i=0; i< fi.nOutput(); i++) ad_K[i] = mkparam(Known(0,i));
    
    adouble ad_partial;
    ad_partial = ad_partial + fi.CalcErr(ad_param, ad_I, ad_K)/nExamples; 
    
    double err;
    ad_partial >>= err;
    
    trace_off();

    delete [] ad_param;
    delete [] ad_I;
    delete [] ad_K;
    
    if ( warn ) {
      size_t counts[1400];
      tapestats(tag,counts);
      
      if (counts[4] > TBUFSIZE) {
	  fprintf(stderr,"tape size is %lu\n",counts[4]);
	  fprintf(stderr,"WARNING: ADOLC compiled TBUFSIZE is too small.\n");
	  fprintf(stderr,"ADOLC does not let me know if .adolcrc is correctly sized\n");
	  fprintf(stderr, "Change ./.adolcrc to increase gradient performance\n");
	  size_t val;
	  for(val=TBUFSIZE; val < counts[4]; val += TBUFSIZE)
	    ;
	  fprintf(stderr,"suggest:\n\"OBUFSIZE\" \"%lu\"\n\"LBUFSIZE\" \"%lu\"\n\"VBUFSIZE\" \"%lu\"\n\"TBUFSIZE\" \"%lu\"\n",val,val,val,val);
	}
    }
  }
  
  void parallel_createAdolcTape()
  {
    // create tape
#pragma omp parallel firstprivate(ADOLC_OpenMP_Handler)
    {
      uint32_t nExamples = Input.rows();
#ifdef _OPENMP
      int nThread=omp_get_num_threads();
      int tid=omp_get_thread_num();
#else
      int nThread=1;
      int tid=0;
#endif
      int tag=tid+1;
      createAdolcTape(tag, (tag==1)?true:false);
#pragma omp barrier
    }
  }
#endif
  
public:
  int devID, warpSize;
  int multiProcessorCount;
  int maxThreadsPerBlock;
  double myErr;
  ObjFcn<REAL_T,myFcnInterest > *d_oFunc;
  
  const char* name() { return "Least Means Squared"; }
  
  FCN_ATTRIBUTES
  ObjFcn(uint32_t nExamples)
  {
    Input.reserve(nExamples, fi.nInput() );
    Known.reserve(nExamples, fi.nOutput() );
    param = new REAL_T[fi.nParam()];
    devID = -1;
    d_oFunc = NULL;
    have_tape=false;
  }

  ~ObjFcn()
  {
    if(devID < 0) { // host object
      delete [] param;
    } else { // have to delete under the covers CUDA allocations
      cudaSetDevice(devID);
      cudaFree(*Input.getDataPtrAddr());
      cudaFree(*Known.getDataPtrAddr());
      cudaFree(param);
      cudaFree(d_oFunc);
    }
  }
  
  __host__
  void offloadParam(REAL_T *h_param)
  {
    if(devID < 0) {
      memcpy(param, h_param, sizeof(REAL_T)*fi.nParam());
    } else {
      cudaSetDevice(devID);
      cudaMemcpy(param, h_param, sizeof(REAL_T)*fi.nParam(),
		 cudaMemcpyHostToDevice);
    }
  }

  __host__
  ObjFcn(ObjFcn &host, int id)
  {

    if(id < 0) {
      std::cerr << "Host devID constructor not implemented" << std::endl;
      exit(1);
    } 
    // only can use host ObjFcn
    assert(host.devID < 0);

    // copy contents of the host oFunc here
    memcpy( this, &host, sizeof(*this));

    *this = host;
    devID = id;
    cudaSetDevice(devID);

    REAL_T *h_Input = *host.Input.getDataPtrAddr();
    REAL_T *h_Known = *host.Known.getDataPtrAddr();

    uint32_t sizeInput= Input.rows() * Input.cols() * sizeof(REAL_T);
    uint32_t sizeKnown= Known.rows() * Known.cols() * sizeof(REAL_T);

    // malloc device data
    cudaMalloc(Input.getDataPtrAddr(), sizeInput);
    cudaMalloc(Known.getDataPtrAddr(), sizeKnown);
    cudaMalloc(&param, sizeof(REAL_T) * nParam() );

    // copy data
    cudaMemcpy(*Input.getDataPtrAddr(), h_Input, sizeInput,
	       cudaMemcpyHostToDevice);
    cudaMemcpy(*Known.getDataPtrAddr(), h_Known, sizeKnown, 
	       cudaMemcpyHostToDevice);

    // now put class on device
    cudaMalloc(&d_oFunc, sizeof(*this));
    cudaMemcpy(d_oFunc, this, sizeof(*this), cudaMemcpyHostToDevice);
  }
  
FCN_ATTRIBUTES
  inline REAL_T *Param() {return param;}

FCN_ATTRIBUTES
  inline REAL_T nParam() {return fi.nParam();}
  
  __device__
  inline void cuda_func()
  {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ double ssum[WARP_SIZE];
 
    if(tid==0) myErr=0.;
    
    if(threadIdx.x < WARP_SIZE)
      ssum[threadIdx.x] = 0.;
    
    const float *I = *Input.getDataPtrAddr();
    const float *K = *Known.getDataPtrAddr();
    register double partial=0.;
    while(tid < Input.rows() ) {
      partial += fi.CalcOpt(param, &I[tid* fi.nInput()], &K[tid* fi.nOutput()]); 
      //partial += fi.CalcOpt(tid, param, &Input, &Known); 
      //double d=fi.CalcOpt(tid, param, &Input, &Known); 
      //partial += d*d;
      //partial += 1;
      tid += blockDim.x * gridDim.x;
    }
    // sum all the partials on each multiprocessor into shared memory
    ssum[threadIdx.x & (WARP_SIZE-1)] += partial;
    __syncthreads();
    
    tid = blockIdx.x*blockDim.x + threadIdx.x;
    volatile double *smem = ssum;
    // sum all threads in each multiprocessor into a single value
    if(threadIdx.x < 16) smem[threadIdx.x] += smem[threadIdx.x + 16];
    if(threadIdx.x < 8) smem[threadIdx.x] += smem[threadIdx.x + 8];
    if(threadIdx.x < 4) smem[threadIdx.x] += smem[threadIdx.x + 4];
    if(threadIdx.x < 2) smem[threadIdx.x] += smem[threadIdx.x + 2];
    if(threadIdx.x < 1) smem[threadIdx.x] += smem[threadIdx.x + 1];
    
    // each thread puts its local sum into shared memory
#if __CUDA_ARCH__ >= 600
#warning "Using Pascal 64-bit atomic add"
    if(threadIdx.x == 0) atomicAdd(&myErr, smem[0]);
#else
    if(threadIdx.x == 0) cuda_atomicAdd(&myErr, smem[0]);
#endif
  }

__host__
  inline double func()
  {
    uint32_t nExamples = Input.rows();
    const float *I = *Input.getDataPtrAddr();
    const float *K = *Known.getDataPtrAddr();
    
    assert(nExamples > 0);

    double err=0.;
#pragma omp parallel for reduction(+:err)
#pragma vector aligned
    for(int i=0; i < nExamples; ++i) {
      err += fi.CalcOpt(param, &I[i* fi.nInput()], &K[i* fi.nOutput()]); 
    }
    return err/nExamples;
  }

  //I don't understand if creating the tapes in parallel once is safe, but it works;
#define CREATE_TAPES_ONCE
  // Use this if errors occur
  //#define CREATE_TAPES_ALWAYS 
  void gen_grad(double *grad, const double *param)
  {
#ifdef USE_GRAD

#ifdef CREATE_TAPES_ONCE
      if(have_tape==false) {
	parallel_createAdolcTape();
	have_tape=true;
      }
#endif

#pragma omp parallel firstprivate(ADOLC_OpenMP_Handler)
    {
      uint32_t nExamples = Input.rows();
#ifdef _OPENMP
      int nThread=omp_get_num_threads();
      int tid=omp_get_thread_num();
#else
      int nThread=1;
      int tid=0;
#endif
      int tag=tid+1;
      if(tid < nExamples) {

#ifdef CREATE_TAPES_ALWAYS
	static int warn_user=true;
	createAdolcTape(tag, warn_user);
	warn_user=false;
#endif

	double *adolGrad = new double[fi.nParam()];
	// set the next input values
	int nInput=Input.cols();
	int nOutput=Known.cols();
	double *newData = new double[nInput+nOutput];

	for(int ex=tid; ex < nExamples; ex += nThread) {
	  int index=0;
	  for(int i=0; i < nInput; i++, index++) newData[index] = Input(ex,i);
	  for(int i=0; i < nOutput; i++, index++) newData[index] = Known(ex,i);
	  set_param_vec(tag,nInput+nOutput,newData);
	  if(gradient(tag, fi.nParam(), param, adolGrad) < 0) {
	    fprintf(stderr,"symbolic gradient failure on conditional branch\n");
	    throw "conditional branch in adolc gradient";
	  }
	  for(int i=0; i < fi.nParam(); i++) grad[i] += adolGrad[i];
	}
	delete [] adolGrad;
	delete [] newData;
      }
#pragma omp barrier
    }
#endif
  }

  FCN_ATTRIBUTES
  inline void pred(Matrix<REAL_T> *Output)
  {
    uint32_t nExamples = Input.rows();
    assert(nExamples > 0);
    assert(Input.rows() == Output->rows() );

    float *I = *Input.getDataPtrAddr();
    float *O = *(Output->getDataPtrAddr());

#pragma omp parallel for 
    for(int i=0; i < nExamples; ++i) {
      fi.CalcOutput(param, &I[i* fi.nInput()], &O[i* fi.nOutput()]); 
    }
  }

FCN_ATTRIBUTES
  inline REAL_T& InputExample(uint32_t i, uint32_t j) { return (Input(i,j)); }
FCN_ATTRIBUTES
  inline REAL_T& KnownExample(uint32_t i, uint32_t j) { return (Known(i,j)); }
FCN_ATTRIBUTES
  inline const char* FcnInterest_name() { return fi.name(); }
FCN_ATTRIBUTES
  inline const char* FcnInterest_gFcnName() { return fi.gFcnName(); }
FCN_ATTRIBUTES
  inline uint32_t FcnInterest_nFlops() { return fi.nFlop(); }
FCN_ATTRIBUTES
  inline Matrix<REAL_T>& InputExample() { return Input; }
FCN_ATTRIBUTES
  inline Matrix<REAL_T>& KnownExample() { return Known; }
  
};

#endif
