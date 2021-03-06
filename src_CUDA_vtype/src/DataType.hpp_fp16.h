#ifndef DATATYPE_HPP
#define DATATYPE_HPP
#include <cuda_fp16.h>

#include "Gfcn.h"

#define DATA_TYPE VecType
#define DATA_TYPE_VLEN 2

//Specify the min compute arch
#define MIN_ARCH_MAJOR 6
#define MIN_ARCH_MINOR 0

struct VecType {
public:
  half2 v;

  FCN_ATTRIBUTES VecType() {}
  FCN_ATTRIBUTES VecType(half2 x) : v(x) {}

  __device__
  static inline void convertFromFloatDup(DATA_TYPE *dst, float *src, uint32_t n)
  {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    half2 *pt = (half2 *) dst;
    for(int i=tid; i < n; i += blockDim.x * gridDim.x) {
      pt[i] = __floats2half2_rn(src[i], src[i]);
    }
  }
  __device__
  static inline void convertFromFloat(DATA_TYPE *dst, float *src, uint32_t n)
  {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    for(int i=DATA_TYPE_VLEN*tid; i < n; i += DATA_TYPE_VLEN*blockDim.x * gridDim.x) {
      dst[i/DATA_TYPE_VLEN].v = __floats2half2_rn(src[i], src[i+1]);
    }
  }
  __device__
  static inline VecType fma(VecType c, VecType b, VecType a) {
    return __hfma2(a.v, b.v, c.v);
  }
  __device__ static inline VecType vecG(VecType x) {
#ifdef LINEAR
#warning "using vector LINEAR g"
    return x;
#else
    return (__floats2half2_rn( G(__low2float(x.v)), G(__high2float(x.v))) );
#endif
    //#ifdef ELLIOTT
    // x/(1.|x|)
    //#warning "Using untested eliott vector function"
        //return h2div(x.v, __hadd2(__floats2half2_rn(1.f,1.f), abs(x)) );
    //#elif LOGISTIC
    //#warning "Using untested logistic vector function"
        //return h2div(__floats2half2_rn(1.f,1.f), 
		     //__hadd2(__floats2half2_rn(1.f,1.f), h2exp(__hneg2(x.v)) ) );
    //#else

  }
  //  __device__ static inline VecType abs(VecType a)
    //{
      //return ( (half2) ( ((uint32_t)a.v) & 0x7FFF7FFF) );
    //}
  __device__ static inline VecType sub(VecType a, VecType b)
  {
    return __hsub2(a.v,b.v);
  }
  __device__ static inline VecType mult(VecType a, VecType b)
  {
    return __hmul2(a.v,b.v);
  }
  __device__ static inline float reduce(VecType x)
  {
    return ( __high2float(x.v) + __low2float(x.v) );
  }
};

#endif
