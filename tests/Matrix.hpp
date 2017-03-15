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

public:
  REAL_T* data_; // For CUDA, cannot use vector<>
  FCN_ATTRIBUTES
  inline unsigned rows() { return nRows;}
  
  FCN_ATTRIBUTES
  inline unsigned cols() { return nCols;}
  
  // dislike this but needed for CUDA and offload
  FCN_ATTRIBUTES
  inline REAL_T** getDataPtrAddr() { return(&data_); }
  FCN_ATTRIBUTES
  inline REAL_T* getDataPtr() { return(data_); }
  
  FCN_ATTRIBUTES
  Matrix() : nRows (0) , nCols (0)
  {
    data_ = NULL;
  }
  
  FCN_ATTRIBUTES
  Matrix(unsigned rows, unsigned cols) : nRows (rows) , nCols (cols)
  {
    //assert(rows > 0 && cols > 0);
    data_ = new REAL_T[rows * cols];
    rowOffset=fixedRow=0;
    cerr << "Matrix copyin from rows, cols constructor" << endl;
    //#pragma acc enter data copyin(this)
    //#pragma acc enter data create(data_[0:nRows*nCols])
    cerr << " size is " << (nRows*nCols) << endl;
  }
  
  FCN_ATTRIBUTES
  ~Matrix()
  {
    if(data_) {
    //#pragma acc exit data delete(data_[0:nRows*nCols], this)
      delete[] data_;
    }
  }
  
  FCN_ATTRIBUTES
  void reserve(uint32_t rows, uint32_t cols)
  {
    assert(data_ == NULL); // only implemented for changing from zero size. 
    nRows = rows;
    nCols = cols;
    data_ = new REAL_T[rows * cols];
    rowOffset=fixedRow=0;
    cerr << "Matrix copyin from reserve rows, cols constructor" << endl;
    //#pragma acc enter data copyin(this)
    //#pragma acc enter data create(data_[0:rows*cols])
    cerr << "\tsize is " << (nRows*nCols) << endl;
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
