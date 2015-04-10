//undefine __declspec
#define __declspec(x)

// Rob Farber
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <malloc.h>
#include <math.h>
#include <omp.h>

#define MIC_DEV 0

#define	ALLOC alloc_if(1) free_if(0)
#define	FREE alloc_if(0) free_if(1)
#define	REUSE alloc_if(0) free_if(0)

// Use a struct to pass and get data from the objective function
typedef struct userData { 
  // Data information
  int nExamples;
  __declspec(align(64)) float * restrict example; 
  __declspec(align(64)) float * restrict param;

  // Timing information
  int isWarmup;
  double timeObjFunc;
  int countObjFunc;
  double timeDataLoad;
  double minTime, maxTime;
} userData_t;

// function to measure wall clock time
inline double getTime() { return(omp_get_wtime());}

#pragma offload_attribute (push, target (mic))

#define restruct
#define __device__
#define __host__

#include "myCommon.h"

// This file defines the function to be evaluated
#include "fcn.h"

// The offload objective function
double _objFunc(unsigned int n,  const double * restrict x,
		double * restrict grad, void * restrict my_func_data)
{
  double err=0.;
  userData_t *uData = (userData_t *) my_func_data;

  // convert from double to float for speed
  for(int i=0; i < N_PARAM; i++) uData->param[i]=x[i];
  
  int nExamples = uData->nExamples;
  // compiler workaround
  __declspec(align(64)) const float * restrict example = uData->example;
  __declspec(align(64)) const float * restrict param = uData->param; 
#pragma acc data copyin(param[0:N_PARAM]) pcopyin(example[0:nExamples*EXAMPLE_SIZE])
#pragma offload target(mic:MIC_DEV) in(param:length(N_PARAM) REUSE) \
                                    out(err) in(example:length(0) REUSE)
#ifdef ORIG_LOOP
  {
    err=0.;
    int nGangs = NUM_SMX * NUM_ACTIVE_SMX_QUEUE;
    int vLen = VEC_LEN; // for some reason PGI needs this as a variable
#pragma acc parallel loop num_gangs(nGangs) vector_length(vLen) reduction(+:err)
  for(int i=0; i < nExamples; ++i) {
    float d=myFunc(i, param, example, nExamples, NULL);
    err += d*d;
    }
  }
#elif MULTICORE_LAYOUT
 {
    err=0.;
    
    int nGangs = NUM_SMX * NUM_ACTIVE_SMX_QUEUE;
    int nThreads= nGangs * VEC_LEN;
    int vLen = VEC_LEN; // for some reason PGI needs this as a variable
    int nExPerThread = nExamples/nThreads;
#pragma acc parallel loop num_gangs(nGangs) vector_length(vLen) reduction(+:err)
    for(int i=0; i < nThreads; ++i) {
      double partial=0.;
      int exEnd = i*nExPerThread + nExPerThread;
      exEnd = (exEnd < nExamples)?exEnd:nExamples;
#pragma acc loop sequential
      for(int j=i*nExPerThread; j < exEnd; ++j) {
	float d=myFunc(j, param, example, nExamples, NULL);
	partial += d*d;
      }
      err += partial;
    }
  }
#elif CALL_CUDA
{
  extern double cuda_objFunc(float*, float *, int, double *);
  double out[1];
#pragma acc data pcreate(out[0:1])
#pragma acc host_data use_device(out, example, param)
  {
    err = cuda_objFunc(param, example, nExamples, out); // initializes out to zero
  }
}
#else
  {
    err=0.;
    
    int nGangs = NUM_SMX * NUM_ACTIVE_SMX_QUEUE;
    int nThreads= nGangs * VEC_LEN;
    int vLen = VEC_LEN; // for some reason PGI needs this as a variable
    #pragma acc parallel loop num_gangs(nGangs) vector_length(vLen) reduction(+:err)
    for(int i=0; i < nThreads; ++i) {
      double partial=0.;
#pragma acc loop sequential
      for(int j=i; j < nExamples ; j += nThreads) {
	float d=myFunc(j, param, example, nExamples, NULL);
	partial += d*d;
      }
      err += partial;
    }
  }
#endif

  return sqrt(err);
}
#pragma offload_attribute (pop)

// The optizimation library callable objective function that gathers timing information
double objFunc(unsigned int n,  const double * restrict x, 
	       double * restrict grad, void * restrict my_func_data)
{
  if(grad) {
    fprintf(stderr,"Gradient not implemented!\n");
    exit(1);
  }
  
  userData_t *uData = (userData_t *) my_func_data;

  double runTime=getTime();
  double err =  _objFunc(n,x,grad,my_func_data);
  runTime = getTime() - runTime;

  if(!uData->isWarmup) {
    // Note a maxTime of zero means this is the first call 
    if(uData->maxTime == 0.) {
      uData->maxTime = uData->minTime = runTime;
    }
    uData->maxTime = (uData->maxTime > runTime)?uData->maxTime:runTime;
    uData->minTime = (uData->minTime < runTime)?uData->minTime:runTime;
    
    uData->timeObjFunc += runTime;
    uData->countObjFunc++;
  }

  return( err );
}

// Called to free memory and report timing information
void fini(userData_t *uData)
{
  int nThreads=0;
  // Intel recommended way to get the number of threads in offload mode.
#pragma offload target(mic:MIC_DEV) out(nThreads)
  {
#pragma omp parallel
    {
#pragma omp single
      {
	nThreads = omp_get_num_threads();
      }
    }
  }
  // Ouput some information
  if(!uData->isWarmup) {
    printf("number OMP threads %d\n", nThreads);
    printf("DataLoadTime %g\n", uData->timeDataLoad);
    printf("AveObjTime %g, countObjFunc %d, totalObjTime %g\n",
	   uData->timeObjFunc/uData->countObjFunc, uData->countObjFunc, uData->timeObjFunc);
#ifdef FLOP_ESTIMATE
    printf("Estimated flops in myFunc %d, estimated average GFlop/s %g\n", FLOP_ESTIMATE,
	   (((double)uData->nExamples*FLOP_ESTIMATE)/(uData->timeObjFunc/uData->countObjFunc)/1.e9) );
    printf("Estimated maximum GFlop/s %g, minimum GFLop/s %g\n",
	   (((double)uData->nExamples*FLOP_ESTIMATE)/(uData->minTime)/1.e9),
	   (((double)uData->nExamples*FLOP_ESTIMATE)/(uData->maxTime)/1.e9) );
  }
#endif

  // free if using offload mode
  __declspec(align(64)) float * restrict example = uData->example;// compiler workaround
  __declspec(align(64)) float * restrict param = uData->param;// compiler workaround
#pragma offload target(mic:MIC_DEV) in(example: length(0) FREE) in(param : length(0) FREE)
  {} 

  // free on the host
  if(uData->example) free(uData->example); uData->example=NULL;
  if(uData->param) free(uData->param); uData->param=NULL;
}

void offloadData(userData_t *uData)
{
#ifdef __INTEL_OFFLOAD
  int nDevices =_Offload_number_of_devices();

  if(nDevices == 0) {
    fprintf(stderr,"No devices found!\n");
    exit -1;
  }

  // If necessary, perform offload transfer and allocation
  double startOffload=getTime();
  __declspec(align(64)) float * restrict example = uData->example; // compiler workaround
  __declspec(align(64)) float * restrict param = uData->param; // compiler workaround
  int Xsiz = uData->nExamples*EXAMPLE_SIZE; // compiler workaround
  // Note: the in for param just allocates memory on the device
#pragma offload target(mic:MIC_DEV) in(example: length(Xsiz) ALLOC) in(param : length(N_PARAM) ALLOC)
  {} 
  // set data load time if using offload mode
  uData->timeDataLoad = getTime() - startOffload;
#endif
}

// loads the binary file of the form:
//    nInput, nOutput, nExamples
//    Input [0] [0:nExamples]
//    Input [1] [0:nExamples]
//    ...
//    Output [0] [0:nExamples]
//    Output [1] [0:nExamples]
//    ...
void init(char*filename, userData_t *uData)
{
  FILE *fn=stdin;

  // check if reading from stdin
  if(strcmp("-", filename) != 0)
    fn=fopen(filename,"r");

  if(!fn) {
    fprintf(stderr,"Cannot open %s\n",filename);
    exit(1);
  }

  // read the header information
  double startTime=getTime();
  int32_t nInput, nOutput;
  int32_t nExamples;
  fread(&nInput,sizeof(int32_t), 1, fn);
  if(nInput != N_INPUT) {
    fprintf(stderr,"Number of inputs incorrect!\n");
    exit(1);
  }
  fread(&nOutput,sizeof(int32_t), 1, fn);
  if(nOutput != N_OUTPUT) {
    fprintf(stderr,"Number of outputs incorrect!\n");
    exit(1);
  }
  fread(&nExamples,sizeof(int32_t), 1, fn);
  if(nExamples <= 0) {
    fprintf(stderr,"Number of examples incorrect!\n");
    exit(1);
  }
  uData->nExamples = nExamples;
  
  // aligned allocation of the data
  uData->example=(float*) memalign(64,nExamples*EXAMPLE_SIZE*sizeof(float));
  if(!uData->example) {
    fprintf(stderr,"Not enough memory for examples!\n");
    exit(1);
  }
  // aligned allocation of the on-device parameters
  uData->param=(float*) memalign(64,N_PARAM*sizeof(float));
  if(!uData->param) {
    fprintf(stderr,"Not enough memory for the parameters!\n");
    exit(1);
  }

  // read the data
  for(int exIndex=0; exIndex < uData->nExamples; exIndex++) {
    for(int i=0; i < nInput; i++) 
      fread(&uData->example[IN(i,uData->nExamples, exIndex)],1, sizeof(float), fn);
    for(int i=0; i < nOutput; i++) 
      fread(&uData->example[OUT(i,uData->nExamples, exIndex)],1, sizeof(float), fn);
  }
  
  // offload the data
  double startOffload=getTime();
  __declspec(align(64)) float * restrict example = uData->example; // compiler workaround
  __declspec(align(64)) float * restrict param = uData->param; // compiler workaround
  int Xsiz = uData->nExamples*EXAMPLE_SIZE; // compiler workaround
  // Note: the in just allocates memory on the device
#pragma offload target(mic:MIC_DEV) in(example: length(Xsiz) ALLOC) in(param : length(N_PARAM) ALLOC)
  {} 
#pragma acc enter data copyin(example[0:Xsiz])

  uData->timeDataLoad = getTime() - startTime;

  if(fn!=stdin) fclose(fn);
}

