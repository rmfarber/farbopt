download nlopt (http://ab-initio.mit.edu/wiki/index.php/NLopt) and
build it here. 

To build with defaults using gcc for OpenMP and CUDA:

   wget http://ab-initio.mit.edu/nlopt/nlopt-2.4.2.tar.gz
   tar -xzf nlopt-2.4.2.tar.gz
   cd nlopt-2.4.2/
   export CFLAGS="-O3"
   export CXXFLAGS="-O3"
   ./configure --prefix=`pwd`/../../install_gcc
   make -j 8 install

This will install in the install_gcc library two levels up. This
directory structure is convenient for benchmarking as it keeps
everything self-contained even though it is not a good idea for
production code.

Instructions to build for Intel Xeon Phi (native and offload) are at:
http://www.drdobbs.com/parallel/numerical-and-computational-optimization/240151128

