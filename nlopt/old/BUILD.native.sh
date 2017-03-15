#download nlopt (http://ab-initio.mit.edu/wiki/index.php/NLopt) and
#build it here. 

   wget http://ab-initio.mit.edu/nlopt/nlopt-2.4.2.tar.gz
   tar -xzf nlopt-2.4.2.tar.gz
   cd nlopt-2.4.2/
   export CC=icc
   export CFLAGS="-mmic -O3"
   export CXX=icpc
   export CXXFLAGS="-mmic -O3"
   export LDFLAGS="-lirc -limf -lsvml"
   ./configure --host=x86 --prefix=`pwd`/../../install_phi
   make -j 8 install
