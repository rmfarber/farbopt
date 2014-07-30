This is a collection of tests and implementations for the Farber
machine-learning and numerical optimization framework. 

CUDA developers will be interested in the following directories 

1) cuda_test: This runs a quick self-contained timing test using an
objective function. Just cd to the directory, sh BUILD, and run the a.out

   ./a.out 0 10 60000000

The rest of the directories use the free nlopt library for nonlinear
optimization found at
http://ab-initio.mit.edu/wiki/index.php/NLopt. 

The current 2-3 release can be retrieved and built. It is assumed that
the library will be installed in the users home directory in
install_cuda via:

  $ ./configure --prefix=$HOME/install_cuda

The CUDA related versions are located in: pca_cuda and nlpca_cuda. The
only difference from a coding point of view is the define in the
BUILD_CUDA script.
   cd pca_cuda
   sh BUILD_CUDA
   sh RUN_CUDA

or
   cd nlpca_cuda
   sh BUILD_CUDA
   sh RUN_CUDA

All the CUDA specific code is in myFunc.h, which is identical between
pca_cuda and nlpca_cuda. The python script genFunc.py can be used to
generate functions for optimization of various complexity, or the
programmer can provide thier own.


Note that this version is limited to 30 seconds of runtime. To check
for convergence (which can take awhile), comment out the QUICK_TEST in
line 1 of train.c. Correctness can be see by running "sh DRAW_DATA"
after performing a run that is not limited to 30 seconds. The two
graphs in the .png files should look the same. The *_known.png is the
graph of the train data and the _pred.png shows the results of the
[pca,nlpca] model that was fit.

Look to my online article for Intel Xeon Phi for more information, and
chapters 2,3, and 10 of my book "CUDA Application Design and
Development"
http://www.drdobbs.com/parallel/numerical-and-computational-optimization/240151128
http://www.drdobbs.com/parallel/getting-to-1-teraflop-on-the-intel-phi-c/240150561

My two 2013 GTC talks discuss many applications of this framework:

S3012: "Simplifying Portable Killer Apps with OpenACC and CUDA-5
Concisely and Efficiently"

   http://nvidia.fullviewmedia.com/gtc2013/0321-MB3-S3012.html


S3443: "Clicking GPUs into a Portable, Persistent and Scalable Massive
Data Framework"

   http://nvidia.fullviewmedia.com/gtc2013/0321-230B-S3443.html

