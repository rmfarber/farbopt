/*
Passing variables / arrays between cython and cpp
Example from 
http://docs.cython.org/src/userguide/wrapping_CPlusPlus.html

Adapted to include passing of multidimensional arrays

*/

#include <vector>
#include <iostream>
#include "FcnOfInterest_config.h"
#include "FcnOfInterest.hpp"

namespace farbopt {
    class PredFcn {
    private:
      std::vector<float> param;
      void loadParam(const char *s);
    public:
      FcnOfInterest<float> fi;
      PredFcn(const char *s);
      ~PredFcn();
      std::vector< float > predict(std::vector< float > sv);
    };
}
