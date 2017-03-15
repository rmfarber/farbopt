#ifndef OBJFCN_HPP
#define OBJFCN_HPP

#include <cstring>
#include <cassert>

#ifndef FCN_ATTRIBUTES
#define FCN_ATTRIBUTES
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
      double d=fi.CalcOpt(i, param, &Input, &Known); 
      err += d*d;
    }
    return err;
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
