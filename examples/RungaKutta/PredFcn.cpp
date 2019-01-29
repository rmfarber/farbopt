/*
Passing variables / arrays between cython and cpp
Example from 
http://docs.cython.org/src/userguide/wrapping_CPlusPlus.html
*/

#include "common.h"
#include "PredFcn.h"

#ifndef PREDFCN
#define PREDFCN generic_fcn
#endif



using namespace farbopt;

void PredFcn::loadParam(const char * filename)
{
  FILE *fn=fopen(filename,"r");
  if(!fn) throw std::runtime_error("Cannot open file");
  
  int parmInFile;
  int ret;
  
  ret=fread(&parmInFile,sizeof(uint32_t), 1, fn);
  if(ret != 1) throw std::runtime_error("header read failure in parameter file");
  
  if(parmInFile != fi.nParam()) {
    if(ret != fi.nParam())
      throw std::runtime_error("Incorrect number of parameters in file");
  }
  param.reserve(fi.nParam());
  ret=fread(&param[0],sizeof(float), fi.nParam(), fn);
  if(ret != fi.nParam()) throw std::runtime_error("parameter read failed");
}

PredFcn::PredFcn(const char *s)
{
  std::cout << "reading params from " << s << std::endl;
  loadParam(s);
}

PredFcn::~PredFcn()
{
}

std::vector< float > PredFcn::predict(std::vector< float > input)
{
  std::vector< float > output;
  if(fi.nOutput() == 0) output.resize(fi.nInput());
  else output.resize(fi.nOutput());
  
  fi.PREDFCN<true,float>(&param[0],&input[0], &output[0]);
  return output;
  
}
