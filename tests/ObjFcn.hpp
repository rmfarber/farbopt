#ifndef OBJFCN_HPP
#define OBJFCN_HPP

#include <cstring>
#include <cassert>

#ifndef FCN_ATTRIBUTES
#define FCN_ATTRIBUTES
#endif

template< typename REAL_T>
class ObjFcn {
private:
  
public:
  int devID;
  double myErr;
  Matrix<REAL_T> Input, Known; 
  
  const char* name() { return "Least Means Squared"; }
  
  FCN_ATTRIBUTES
  ObjFcn(uint32_t nExamples, int colsInput, int colsKnown)
  {
    Input.reserve(nExamples, colsInput );
    Known.reserve(nExamples, colsKnown );
    //#pragma acc enter data copyin(this)
  }

  ~ObjFcn()
  {
    //#pragma acc exit data delete(this)
  }
  
FCN_ATTRIBUTES
  inline REAL_T& InputExample(uint32_t i, uint32_t j) { return (Input(i,j)); }
FCN_ATTRIBUTES
  inline REAL_T& KnownExample(uint32_t i, uint32_t j) { return (Known(i,j)); }
FCN_ATTRIBUTES
  inline Matrix<REAL_T>& InputExample() { return Input; }
FCN_ATTRIBUTES
  inline Matrix<REAL_T>& KnownExample() { return Known; }
  
};

#endif
