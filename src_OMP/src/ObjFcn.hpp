#ifndef OBJFCN_HPP
#define OBJFCN_HPP

#include <cstring>
#include <cassert>

#ifndef FCN_ATTRIBUTES
#define FCN_ATTRIBUTES
#endif

#ifdef USE_GRAD
#include <adolc/adolc.h>
#ifdef _OPENMP
#include <omp.h>
#include <adolc/adolc_openmp.h>
#endif
#endif


template< typename REAL_T, typename myFcnInterest >
class ObjFcn {
private:
  myFcnInterest fi;
  Matrix<REAL_T> Input, Known; 
  REAL_T *param;
  ObjFcn() { }
  
public:
  int devID;
  double myErr;
  
  const char* name() { return "Least Means Squared"; }
  
  FCN_ATTRIBUTES
  ObjFcn(uint32_t nExamples)
  {
    Input.reserve(nExamples, fi.nInput() );
    Known.reserve(nExamples, fi.nOutput() );
    //param = new REAL_T[fi.nParam()];
    param = (REAL_T *) aligned_alloc(64,sizeof(REAL_T)*nParam());
    devID = -1;
  }

  ~ObjFcn()
  {
    if(devID < 0) { // host object
      //delete [] param;
      free(param);
    }
  }
  
  void offloadParam(REAL_T *h_param)
  {
    if(devID < 0) {
      memcpy(param, h_param, sizeof(REAL_T)*fi.nParam());
    } 
  }

  ObjFcn(ObjFcn &host, int id)
  {

    std::cerr << "Host devID copy constructor not implemented" << std::endl;
    exit(1);
  }
  
FCN_ATTRIBUTES
  inline REAL_T *Param() {return param;}

FCN_ATTRIBUTES
  inline uint32_t nParam() {return fi.nParam();}
  
  inline double func()
  {
    uint32_t nExamples = Input.rows();
    double err=0.;
    
    assert(nExamples > 0);

#pragma omp parallel for reduction(+:err)
#pragma vector aligned
    //#pragma omp parallel for reduction(+:err)
    for(int i=0; i < nExamples; ++i) {
      err += fi.CalcOpt(i, param, &Input, &Known); 
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

  void gen_grad(double *grad, const double *param)
  {
#ifdef USE_GRAD

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
	trace_on(tag,1);
	adouble *ad_param = new adouble[fi.nParam()];
	for (int i=0; i< fi.nParam(); i++) {
	  ad_param[i] <<= param[i];
	}
	
	adouble ad_partial;
	ad_partial = ad_partial + fi.ad_fcn(tid, ad_param, &Input, &Known)/nExamples; 
	
	double err;
	ad_partial >>= err;
	
	trace_off();
	
	double *adolGrad = new double[fi.nParam()];
	reverse(tag,1,N_PARAM,0,1.0,adolGrad);
	
	// set the next input values
	int nInput=Input.cols();
	int nOutput=Known.cols();
	double *newData = new double[nInput+nOutput];
      
	for(int ex=tid; ex < nExamples; ex += nThread) {
	  int index=0;
	  for(int i=0; i < nInput; i++, index++) newData[index] = Input(ex,i);
	  for(int i=0; i < nOutput; i++, index++) newData[index] = Known(ex,i);
	  set_param_vec(tag,nInput+nOutput,newData);
	  zos_forward(tag,1,N_PARAM,1,param,&err);
	  reverse(tag,1,N_PARAM,0,1.0,adolGrad);
	  
	  for(int i=0; i < fi.nParam(); i++) grad[i] += adolGrad[i];
	  
	  //#pragma critical
	  	  //{
	    	  //float check_p[N_PARAM];
	    	  //for(int i=0; i < N_PARAM; i++) check_p[i] = param[i];
	    	  //double check= fi.CalcOpt(ex, check_p, &Input, &Known)/nExamples; 
	    	  //printf("zos_forward %f, func %f\n",err, check);
	  	  //}
	}
	delete [] adolGrad;
	delete [] ad_param;
	delete [] newData;
      }
#pragma omp barrier
    }
#endif
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
