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
  bool have_tape;
  
#ifdef USE_GRAD
  void createAdolcTape(int tag, bool warn)
  {
    uint32_t nExamples = Input.rows();

    for(int i=0; i < fi.nParam(); i++) param[i]=0;
    trace_on(tag,1);
    adouble *ad_param = new adouble[fi.nParam()];
    for (int i=0; i< fi.nParam(); i++) {
      ad_param[i] <<= param[i];
    }
    
    adouble ad_partial;
    ad_partial = ad_partial + fi.ad_fcn(0, ad_param, &Input, &Known)/nExamples; 
    
    double err;
    ad_partial >>= err;
    
    trace_off();
    
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
    }
#pragma omp barrier
  }
#endif

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
    have_tape=false;
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
    const uint32_t nExamples = Input.rows();
    const uint32_t nOutput = (N_OUTPUT==0)?N_INPUT:N_OUTPUT; //special case autoencoders
    const float *I = *Input.getDataPtrAddr();
    const float *K = *Known.getDataPtrAddr();
    
    assert(nExamples > 0);

    double err=0.;
#pragma omp parallel for reduction(+:err)
#pragma vector aligned
    for(int i=0; i < nExamples; ++i) {
      err += fi.CalcOpt(param, &I[i*N_INPUT], &K[i*nOutput]); 
    }
    return err/nExamples;
  }

  FCN_ATTRIBUTES
  inline void pred(Matrix<REAL_T> *Output)
  {
    const uint32_t nExamples = Input.rows();
    const uint32_t nOutput = (N_OUTPUT==0)?N_INPUT:N_OUTPUT; //special case autoencoders
    assert(nExamples > 0);
    assert(Input.rows() == Output->rows() );

    float *I = *Input.getDataPtrAddr();
    float *O = *(Output->getDataPtrAddr());

#pragma omp parallel for 
    for(int i=0; i < nExamples; ++i) {
      fi.CalcOutput(param, &I[i*N_INPUT], &O[i*nOutput]); 
    }
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
	  if(gradient(tag, N_PARAM, param, adolGrad) < 0) {
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
