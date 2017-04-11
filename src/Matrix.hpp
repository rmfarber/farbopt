#ifndef MATRIX_HPP
#define MATRIX_HPP

#ifndef FCN_ATTRIBUTES
#define FCN_ATTRIBUTES
#endif

#include <cassert>

template< typename REAL_T>
struct Matrix {
private:
  unsigned nRows, nCols, fixedRow, rowOffset;
  REAL_T* data_; // For CUDA, cannot use vector<> 
  
public:

  FCN_ATTRIBUTES
  inline unsigned rows() { return nRows;}
  
  FCN_ATTRIBUTES
  inline unsigned cols() { return nCols;}
  
  // dislike this but needed for CUDA and offload
  FCN_ATTRIBUTES
  inline REAL_T** getDataPtrAddr() { return(&data_); }
  
  FCN_ATTRIBUTES
  Matrix() : nRows (0) , nCols (0)
  {
    data_ = NULL;
  }
  
  FCN_ATTRIBUTES
  Matrix(unsigned rows, unsigned cols) : nRows (rows) , nCols (cols)
  {
    //assert(rows > 0 && cols > 0);
    //data_ = new REAL_T[rows * cols];
    data_ = (REAL_T *) aligned_alloc(64,sizeof(REAL_T)*nRows*nCols);

    rowOffset=fixedRow=0;
  }
  
  FCN_ATTRIBUTES
  ~Matrix()
  {
    if(data_) {
      free(data_);
    }
  }
  
  FCN_ATTRIBUTES
  void reserve(uint32_t rows, uint32_t cols)
  {
    assert(data_ == NULL); // only implemented for changing from zero size. 
    nRows = rows;
    nCols = cols;
    rowOffset=fixedRow=0;
    //data_ = new REAL_T[rows * cols];
    data_ = (REAL_T *) aligned_alloc(64,sizeof(REAL_T)*nRows*nCols);
  }
  
  FCN_ATTRIBUTES
  inline REAL_T& operator() (unsigned row, unsigned col)
  {
    assert(row < nRows && col < nCols);
#ifdef LAYOUT_SOA
    return data_[nRows * col + row];
#else
    return data_[nCols * row + col];
#endif
  }
  
  FCN_ATTRIBUTES
  inline REAL_T& operator() (unsigned row, unsigned col) const
  {
    assert(row < nRows && col < nCols);
#ifdef LAYOUT_SOA
    return data_[nRows * col + row];
#else
    return data_[nCols * row + col];
#endif
  }
  
};
#endif
