#ifndef DATATYPE_HPP
#define DATATYPE_HPP

#include "Gfcn.h"

#define DATA_TYPE VecType
#define DATA_TYPE_VLEN 2

//Specify the min compute arch
#define MIN_ARCH_MAJOR 3
#define MIN_ARCH_MINOR 5


struct VecType {
public:
  float2 v;

  FCN_ATTRIBUTES VecType() {}
  FCN_ATTRIBUTES VecType(float2 x) : v(x) {}

  FCN_ATTRIBUTES
  static inline void convertFromFloatDup(DATA_TYPE *dst, float *src, uint32_t n)
  {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    for(int i=tid; i < n; i += blockDim.x * gridDim.x) {
      dst[i] = make_float2(src[i], src[i]);
    }
  }
  FCN_ATTRIBUTES
  static inline void convertFromFloat(DATA_TYPE *dst, float *src, uint32_t n)
  {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    for(int i=DATA_TYPE_VLEN*tid; i < n; i += DATA_TYPE_VLEN * blockDim.x * gridDim.x) {
      dst[i/DATA_TYPE_VLEN] = make_float2(src[i], src[i+1]);
    }
  }
  FCN_ATTRIBUTES static inline VecType fma(VecType c, VecType b, VecType a) {
#ifdef __CUDA_ARCH__
    return make_float2(c.v.x + b.v.x * a.v.x, c.v.y + b.v.y * a.v.y);
    //return make_float2(__fmaf_rn(a.v.x,b.v.x,c.v.x), __fmaf_rn(a.v.y,b.v.y,c.v.y));
#else
    return make_float2(c.v.x + b.v.x * a.v.x, c.v.y + b.v.y * a.v.y);
#endif
  }
  FCN_ATTRIBUTES static inline VecType vecG(VecType a) {
    return make_float2(G(a.v.x), G(a.v.y));
  }
  FCN_ATTRIBUTES static inline VecType sub(VecType a, VecType b)
  {
    return make_float2(a.v.x - b.v.x, a.v.y - b.v.y);
  }
  FCN_ATTRIBUTES static inline VecType mult(VecType a, VecType b)
  {
    return make_float2(a.v.x * b.v.x, a.v.y * b.v.y);
  }
  FCN_ATTRIBUTES static inline float reduce(VecType arg)
  {
    return ( arg.v.x + arg.v.y);
  }
};

#endif
