#ifndef OBJFCN_HPP
#define OBJFCN_HPP


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cassert>

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
    
    register double partial=0.;
    while(tid < Input.rows() ) {
      partial += fi.CalcOpt(tid, param, &Input, &Known); 
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
    double err=0.;
    
    assert(nExamples > 0);

#pragma omp parallel for reduction(+:err)
    for(int i=0; i < nExamples; ++i) {
      err +=fi.CalcOpt(i, param, &Input, &Known); 
    }
    return err/nExamples;
  }

  FCN_ATTRIBUTES
  inline void pred(Matrix<REAL_T> *Output)
  {
    uint32_t nExamples = Input.rows();
    assert(nExamples > 0);
    assert(Input.rows() == Output->rows() );

#pragma omp parallel for 
    for(int i=0; i < nExamples; ++i) {
      fi.CalcOutput(i, param, &Input, Output); 
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
