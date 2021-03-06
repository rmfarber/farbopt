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
  int foo; // placeholder so OpenACC runtime does not confuse Input and ObjFunc
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
    int nParam = fi.nParam();
    param = new REAL_T[nParam];
    devID = -1;
#pragma acc enter data copyin(this)
#pragma acc enter data create(param[0:nParam])
    REAL_T **ppt_input = Input.getDataPtrAddr();
    REAL_T **ppt_known = Known.getDataPtrAddr();
    int sizeKnown = nExamples * fi.nOutput();
    int sizeInput = nExamples * fi.nInput();
#pragma acc enter data create(ppt_input[0:sizeInput])
#pragma acc enter data create(ppt_known[0:sizeKnown])
  }

  ~ObjFcn()
  {
    if(devID < 0) { // host object
      delete [] param;
    }
  }
  
  void offloadParam(REAL_T *h_param)
  {
    if(devID < 0) {
      int nParam = fi.nParam();
      memcpy(param, h_param, sizeof(REAL_T)*nParam);
#pragma acc update device(param[0:nParam])
    } 
  }

  ObjFcn(ObjFcn &host, int id)
  {

    cerr << "Host devID copy constructor not implemented" << endl;
    exit(1);
  }
  
FCN_ATTRIBUTES
  inline REAL_T *Param() {return param;}

FCN_ATTRIBUTES
  inline REAL_T nParam() {return fi.nParam();}
  
  inline double func()
  {
    uint32_t nExamples = Input.rows();
    double err=0.;
    
    assert(nExamples > 0);

#pragma omp parallel for reduction(+:err)
#pragma acc parallel loop reduction(+:err)
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
