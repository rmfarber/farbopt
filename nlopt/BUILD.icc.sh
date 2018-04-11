#download nlopt (http://ab-initio.mit.edu/wiki/index.php/NLopt) and
#build it here. 

#To build with defaults using icc

DIR=nlopt-2.4.2
FILE=$DIR.tar.gz

if test -e $FILE
then
echo "have file"
else
   wget http://ab-initio.mit.edu/nlopt/$FILE
fi

   rm -fr $DIR
   tar -xzf $FILE
   cd $DIR
   cd nlopt-2.4.2/
   export CC=icc
   export CFLAGS="-O3 -xhost"
   export CXX=icpc
   export CXXFLAGS="-O3 -xhost"
   export LDFLAGS="-lirc -limf -lsvml"
   ./configure --host=x86 --prefix=`pwd`/../../lib/nlopt_icc
   make -j 8 install

#This will install in the install_gcc library two levels up. This
#directory structure is convenient for benchmarking as it keeps
#everything self-contained even though it is not a good idea for
#production code.

