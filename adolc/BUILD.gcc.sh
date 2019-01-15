#download adolc (https://www.coin-or.org/download/source/ADOL-C/ADOL-C-2.6.3.tgz) and 
#build it here. 

#To build with defaults using gcc 


DIR=ADOL-C-2.6.3
FILE=ADOL-C-2.6.3.tgz

if test -e $FILE
then
echo "have file"
else
   wget https://www.coin-or.org/download/source/ADOL-C/$FILE
fi
   rm -fr $DIR
   tar -xzf $FILE
   cd $DIR
   export CFLAGS="-O3"
   export CXXFLAGS="-O3"
   ./configure --prefix=`pwd`/../../lib/adolc_gcc --with-openmp-flag=-fopenmp
   make -j 8 install

#This will install in the install_gcc library two levels up. This
#directory structure is convenient for benchmarking as it keeps
#everything self-contained even though it is not a good idea for
#production code.

