#ifndef GFCN_H
#define GFCN_H

#ifndef FCN_ATTRIBUTES
#define FCN_ATTRIBUTES
#endif

#ifdef USE_GRAD
#include <adolc/adolc.h>
#endif
#include <math.h>

// ********************************************************
//                 Activation functions
// ********************************************************
typedef struct {
  FCN_ATTRIBUTES
  inline static const char *name()  {
    return "Elliott activation: x/(1+fabsf(x))";
  }
  FCN_ATTRIBUTES
  inline static int nparam() { return(0);} 
  FCN_ATTRIBUTES
  inline static int nflop() { return(3);} 
  FCN_ATTRIBUTES
  inline static float fcn(float x) { return( x/(1.f+fabsf(x)) ) ;} 
#ifdef USE_GRAD
  FCN_ATTRIBUTES
  inline static adouble fcn(adouble x) { return( x/(1.+fabs(x)) ) ;} 
#endif
} Elliott_G;

typedef struct {
  FCN_ATTRIBUTES
  inline static const char *name()  { return "tanh()"; }
  FCN_ATTRIBUTES
  inline static int nparam() { return(0);} 
  FCN_ATTRIBUTES
  inline static int nflop() { return(7);} 
  FCN_ATTRIBUTES
  inline static float fcn(float x) {
    float t1=expf(x); float t2=expf(-x);
    return( (t1-t2)/(t1+t2) ) ;
  } 
#ifdef USE_GRAD
  FCN_ATTRIBUTES
  inline static adouble fcn(adouble x) { return( tanh(x) ) ;} 
#endif
} Tanh_G;

typedef struct {
  FCN_ATTRIBUTES
  inline static const char *name()  { return "logistic()"; }
  FCN_ATTRIBUTES
  inline static int nparam() { return(0);} 
  FCN_ATTRIBUTES
  inline static int nflop() { return(9);} 
  FCN_ATTRIBUTES
  inline static float fcn(float x) { return( 1.f/(1.f+expf(-x)) ) ;} 
#ifdef USE_GRAD
  FCN_ATTRIBUTES
  inline static adouble fcn(adouble x) { return( 1./(1.+exp(-x)) ) ;} 
#endif
} Logistic_G;

typedef struct {
  FCN_ATTRIBUTES
  inline static const char *name()  {
    return "Exponential linear unit(parameter a locked to 1.)";
  }
  FCN_ATTRIBUTES
  inline static int nparam() { return(0);} 
  FCN_ATTRIBUTES
  inline static int nflop() { return(7);} 
  FCN_ATTRIBUTES
  inline static float fcn(float x) { return( (x>0.f)?x:(expf(x)-1.f) ); } 
#ifdef USE_GRAD
  FCN_ATTRIBUTES
  inline static adouble fcn(adouble x) {adouble tmp; condassign(tmp, x, x, exp(x)-1); return(tmp ); } 
#endif
} Elu_G;

typedef struct {
  FCN_ATTRIBUTES
  inline static const char *name()  { return "Linear"; }
  FCN_ATTRIBUTES
  inline static int nparam() { return(0);} 
  FCN_ATTRIBUTES
  inline static int nflop() { return(0);} 
  FCN_ATTRIBUTES
  inline static float fcn(float x) { return( x ) ;} 
#ifdef USE_GRAD
  FCN_ATTRIBUTES
  inline static adouble fcn(adouble x) { return( x ) ;} 
#endif
} Linear_G;

// ****************************************************
//                        Functions
// ****************************************************

template<int LEN_LAYER> class AllLayer2neuron {
 public:
  FCN_ATTRIBUTES
    inline static const char *name()  { return "AllLayer2neuron"; }
  FCN_ATTRIBUTES
    inline static int nparam() { return(1+LEN_LAYER);} 
  FCN_ATTRIBUTES
    inline static int nflop() { return(1 + 2*LEN_LAYER); } 
  FCN_ATTRIBUTES
    inline static float fcn(float *layer, const float *param, int index) {
    float neuron = param[index++];
    for(int i=0; i < LEN_LAYER; ++i) neuron += layer[i] * param[index++];
    return neuron;
  } 
#ifdef USE_GRAD
  FCN_ATTRIBUTES
    inline static adouble fcn(adouble *layer, const adouble *param, int index) {
    adouble neuron = param[index++];
    for(int i=0; i < LEN_LAYER; ++i) neuron += param[index++]*layer[i];
    return neuron;
  } 
#endif
};

template<int LEN_FROM_LAYER, int LEN_TO_LAYER> class FromAll2all {
 public:
  FCN_ATTRIBUTES
    inline static const char *name()  { return "FromAll2all"; }
  FCN_ATTRIBUTES
    inline static int nparam() { return(LEN_TO_LAYER*LEN_FROM_LAYER);} 
  FCN_ATTRIBUTES
    inline static int nflop() { return(2*LEN_TO_LAYER*LEN_FROM_LAYER);} 
  FCN_ATTRIBUTES
    inline static void fcn(const float *from_layer, float *to_layer,
			   const float *param, int index) {
    for(int from=0; from < LEN_FROM_LAYER; ++from)
      for(int to=0; to < LEN_TO_LAYER; ++to)
	to_layer[to] += param[index++] * from_layer[from];
  } 
#ifdef USE_GRAD
  FCN_ATTRIBUTES
    inline static void fcn(const adouble *from_layer, adouble *to_layer,
			   const adouble *param, int index) {
    for(int from=0; from < LEN_FROM_LAYER; ++from)
      for(int to=0; to < LEN_TO_LAYER; ++to)
	to_layer[to] += param[index++] * from_layer[from];
  } 
#endif
};

template<int LEN_LAYER> class AllLayer_Init {
 public:
  FCN_ATTRIBUTES
    inline static const char *name()  { return "AllLayer_Init"; }
  FCN_ATTRIBUTES
    inline static int nparam() { return(LEN_LAYER);} 
  FCN_ATTRIBUTES
    inline static int nflop() { return(LEN_LAYER);} 
  FCN_ATTRIBUTES
    inline static void fcn(float *layer, const float *param, int index) {
    for(int to=0; to < LEN_LAYER; ++to) layer[to] = param[index++];
  } 
#ifdef USE_GRAD
  FCN_ATTRIBUTES
    inline static void fcn(adouble *layer, const adouble *param, int index) {
    for(int to=0; to < LEN_LAYER; ++to) layer[to] = param[index++];
  } 
#endif
};

template<int LEN_LAYER, typename Functor> class AllLayer_G {
 public:
  FCN_ATTRIBUTES
    inline static const char *name()  { return "AllLayer_G"; }
  FCN_ATTRIBUTES
    inline static int nparam() { return(LEN_LAYER*Functor::nparam());} 
  FCN_ATTRIBUTES
    inline static int nflop() { return(LEN_LAYER*Functor::nflop());} 
  FCN_ATTRIBUTES
    inline static void fcn(float *layer, const float *param, int index) {
    for(int to=0; to < LEN_LAYER; ++to) layer[to] = Functor::fcn(layer[to]);
  } 
#ifdef USE_GRAD
  FCN_ATTRIBUTES
    inline static void fcn(adouble *layer, const adouble *param, int index) {
    for(int to=0; to < LEN_LAYER; ++to) layer[to] = Functor::fcn(layer[to]);
  } 
#endif
};

#endif
