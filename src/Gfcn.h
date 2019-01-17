#ifndef GFCN_H
#define GFCN_H

#ifndef FCN_ATTRIBUTES
#define FCN_ATTRIBUTES ""
#endif

#ifdef USE_GRAD
#include <adolc/adolc.h>
#endif
#include <math.h>

// Define the Sigmoid
#if defined(ELLIOTT_ACTIVATION_FCN)

#define G_DESC_STRING "Elliott activation: x/(1+fabsf(x))"
FCN_ATTRIBUTES
inline float G(float x) { return( x/(1.f+fabsf(x)) ) ;} 
#define G_ESTIMATE 3 // estimate flops for G
#ifdef USE_GRAD
adouble G_ad(adouble x) { return( x/(1.+fabs(x)) ) ;} 
#endif

#elif defined(TANH_ACTIVATION_FCN)

#define G_DESC_STRING "tanh() estimated 7 flops to calc G"
FCN_ATTRIBUTES
inline float G(float x) { return( tanhf(x) ) ;} 
#define G_ESTIMATE 7 // estimate 7 flops for G
#ifdef USE_GRAD
adouble G_ad(adouble x) { return( tanh(x) ) ;} 
#endif

#elif defined(LOGISTIC_ACTIVATION_FCN)

#define G_DESC_STRING "logistic() estimated 9 flops to calc G"
FCN_ATTRIBUTES
inline float G(float x) { return( 1.f/(1.f+expf(-x)) ) ;} 
#define G_ESTIMATE 9 // estimate 7 flops for G
#ifdef USE_GRAD
adouble G_ad(adouble x) { return( 1./(1.+exp(-x)) ) ;} 
#endif

#elif defined(ELU_ACTIVATION_FCN)

#define G_DESC_STRING "Exponential linear unit(parameter a locked to 1.) estimated 7 flops to calc G"
FCN_ATTRIBUTES
inline float G(float x) { return( (x>0.f)?x:(expf(x)-1.f) ); } 
#define G_ESTIMATE 7 // estimate flops for G
#ifdef USE_GRAD
adouble G_ad(adouble x) { return( (x>0.f)?x:(exp(x)-1.f) ); } 
#endif

#else // Use linear

FCN_ATTRIBUTES
#define G_DESC_STRING "LINEAR"
FCN_ATTRIBUTES
inline float G(float x) { return( x ) ;} 
#ifdef USE_GRAD
adouble G_ad(adouble x) { return( x ) ;} 
#endif
#define G_ESTIMATE 0 

#endif

#endif
