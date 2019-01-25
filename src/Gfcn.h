#ifndef GFCN_H
#define GFCN_H

#ifndef FCN_ATTRIBUTES
#define FCN_ATTRIBUTES
#endif

#ifdef USE_GRAD
#include <adolc/adolc.h>
#endif
#include <math.h>

typedef struct {
  FCN_ATTRIBUTES
  inline static const char *name()  {
    return "Elliott activation: x/(1+fabsf(x))";
  }
  FCN_ATTRIBUTES
  inline static int flops() { return(3) ;} 
  FCN_ATTRIBUTES
  inline static float G(float x) { return( x/(1.f+fabsf(x)) ) ;} 
  FCN_ATTRIBUTES
  inline static adouble G(adouble x) { return( x/(1.+fabs(x)) ) ;} 
} Elliott_nn;

typedef struct {
  FCN_ATTRIBUTES
  inline static const char *name()  {
    return "tanh() estimated 7 flops to calc G";
  }
  FCN_ATTRIBUTES
  inline static int flops() { return(7) ;} 
  FCN_ATTRIBUTES
  inline static float G(float x) {
    float t1=expf(x); float t2=expf(-x);
    return( (t1-t2)/(t1+t2) ) ;
  } 
  FCN_ATTRIBUTES
  inline static adouble G(adouble x) { return( tanh(x) ) ;} 
} Tanh_nn;

typedef struct {
  FCN_ATTRIBUTES
  inline static const char *name()  {
    return "logistic() estimated 9 flops to calc G";
  }
  FCN_ATTRIBUTES
  inline static int flops() { return(9) ;} 
  FCN_ATTRIBUTES
  inline static float G(float x) { return( 1.f/(1.f+expf(-x)) ) ;} 
  FCN_ATTRIBUTES
  inline static adouble G(adouble x) { return( 1./(1.+exp(-x)) ) ;} 
} Logistic_nn;

typedef struct {
  FCN_ATTRIBUTES
  inline static const char *name()  {
    return "Exponential linear unit(parameter a locked to 1.) estimated 7 flops";
  }
  FCN_ATTRIBUTES
  inline static int flops() { return(7) ;} 
  FCN_ATTRIBUTES
  inline static float G(float x) { return( (x>0.f)?x:(expf(x)-1.f) ); } 
  FCN_ATTRIBUTES
  inline static adouble G(adouble x) {adouble tmp; condassign(tmp, x, x, exp(x)-1); return(tmp ); } 
} Elu_nn;

typedef struct {
  FCN_ATTRIBUTES
  inline static const char *name()  { return "Linear"; }
  FCN_ATTRIBUTES
  inline static int flops() { return(0);} 
  FCN_ATTRIBUTES
  inline static float G(float x) { return( x ) ;} 
  FCN_ATTRIBUTES
  inline static adouble G(adouble x) { return( x ) ;} 
} Linear_nn;

#endif
