#download nlopt (http://ab-initio.mit.edu/wiki/index.php/NLopt) and
#build it here. 

#To build with defaults using gcc for OpenMP and CUDA:


DIR=nlopt-2.4.2
FILE=$DIR.tar.gz

if test -e $FILE
then
echo "have file"
else
   wget http://ab-initio.mit.edu/nlopt/$FILE
fi

   tar -xzf $FILE
   cd $DIR
   export CFLAGS="-O3"
   export CXXFLAGS="-O3"
   ./configure --prefix=`pwd`/../../lib/nlopt_gcc
   make -j 8 install

#This will install in the install_gcc library two levels up. This
#directory structure is convenient for benchmarking as it keeps
#everything self-contained even though it is not a good idea for
#production code.

