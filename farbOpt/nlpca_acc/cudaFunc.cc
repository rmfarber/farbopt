// Rob Farber

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include "myCommon.h"

__device__
#include "fcn.h"

__device__ inline void atomicAdd (double *address, double value)
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

template <class T, class T1, unsigned int WARP_SIZE>
__global__ void d_objFunc(T* d_param, T *d_example, int nExamples, T1 *out)
{
  __shared__ T1 ssum[WARP_SIZE];
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid==0) *out=0.f;

  if(threadIdx.x < WARP_SIZE)
    ssum[threadIdx.x] = 0.;

  register T1 partial=0.f;
  while(tid < nExamples) {
    T d= myFunc(tid, d_param, d_example, nExamples, NULL);
    partial += d*d;
    //partial += tid;
    tid += blockDim.x * gridDim.x;
  }
  // sum all the partials on each multiprocessor into shared memory
  ssum[threadIdx.x & (WARP_SIZE-1)] += partial;
  __syncthreads();

  tid = blockIdx.x*blockDim.x + threadIdx.x;
  volatile T1 *smem = ssum;
  // sum all threads in each multiprocessor into a single value
  if(threadIdx.x < 16) smem[threadIdx.x] += smem[threadIdx.x + 16];
  if(threadIdx.x < 8) smem[threadIdx.x] += smem[threadIdx.x + 8];
  if(threadIdx.x < 4) smem[threadIdx.x] += smem[threadIdx.x + 4];
  if(threadIdx.x < 2) smem[threadIdx.x] += smem[threadIdx.x + 2];
  if(threadIdx.x < 1) smem[threadIdx.x] += smem[threadIdx.x + 1];

  // each thread puts its local sum into shared memory
  if(threadIdx.x == 0) atomicAdd(out, smem[0]);
}

extern "C" double cuda_objFunc(float* d_param, float *d_example, int nExamples, double *d_out)
{

  d_objFunc<float,double,32><<<NUM_SMX*NUM_ACTIVE_SMX_QUEUE, VEC_LEN>>>
    (d_param, d_example, nExamples,d_out);
  
  cudaError_t ret=cudaGetLastError();
  if( ret != cudaSuccess) {
    fprintf(stderr,"CUDA error: %s\n", cudaGetErrorString(ret));
    exit(-1);
  }

  double error;
  ret=cudaMemcpy(&error,d_out, sizeof(double), cudaMemcpyDefault);
  if( ret != cudaSuccess) {
    fprintf(stderr,"CUDA error: %s\n", cudaGetErrorString(ret));
    exit(-1);
  }
  return error;
}
