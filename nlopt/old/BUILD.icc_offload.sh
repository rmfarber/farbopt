#download nlopt (http://ab-initio.mit.edu/wiki/index.php/NLopt) and
#build it here. 

#For Offload mode:

   wget http://ab-initio.mit.edu/nlopt/nlopt-2.4.2.tar.gz
   tar -xzf nlopt-2.4.2.tar.gz
   cd nlopt-2.4.2/
   export CC=icc
   export CFLAGS="-O3 -xhost"
   export CXX=icpc
   export CXXFLAGS="-O3 -xhost"
   export LDFLAGS="-lirc -limf -lsvml"
   ./configure --host=x86 --prefix=`pwd`/../../install_mic
   make -j 8 install

#For Native mode (assumes the offload mode was just built):
#   export CFLAGS="-mmic -O3"
#   export CXXFLAGS="-mmic -O3"
#   make clean
#   ./configure --host=x86 --prefix=`pwd`/../../install_mic_native
#   make -j 8 install
