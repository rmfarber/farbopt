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
 float *example; 
 float *param;

  // Timing information
  int isWarmup;
  double timeObjFunc;
  int countObjFunc;
  double timeDataLoad;
  double minTime, maxTime;
} userData_t;

// function to measure wall clock time
inline double getTime() { return(omp_get_wtime());}

#pragma omp declare  target

// helper macros to index into the example array
#define IN(i,nExamples,j)  (i*nExamples+j)
#define OUT(i,nExamples,j)  ((i+N_INPUT)*nExamples+j)

// Define the Sigmoid
#ifdef USE_LINEAR
#define G_DESC_STRING "generated_PCA_func LINEAR()"
inline float G(float x) { return( x ) ;} 
#define G_ESTIMATE 0 
#elif USE_TANH
#define G_DESC_STRING "generated_func tanh()"
inline float G(float x) { return( tanhf(x) ) ;} 
#define G_ESTIMATE 7 // estimate 7 flops for G
#elif LOGISTIC
#define G_DESC_STRING "generated func logistic()"
inline float G(float x) { return( 1.f/(1.f+expf(-x)) ) ;} 
#define G_ESTIMATE 7 // estimate flops for G
#else // Use Elliott function
#define G_DESC_STRING "generated func Eliott activation: x/(1+fabsf(x))"
inline float G(float x) { return( x/(1.f+fabsf(x)) ) ;} 
#define G_ESTIMATE 3 // estimate flops for G
#endif

// This file defines the function to be evaluated
#include "fcn.h"

#pragma declare simd uniform(examples) linear(nExamples:1)
double simdObjFunc(int nExamples, float *restrict example, float * restrict param)
{
  double err=0.;
#pragma omp target map(to:param[0:N_PARAM]) map(alloc:example[0:nExamples])
  {
    err=0.; // initialize error here in case offload selected
    
    __assume_aligned(param, 64); __assume_aligned(example, 64);
#pragma omp parallel for simd reduction(+:err)
    for(int i=0; i < nExamples; i++) {
      float d=myFunc(i, param, example, nExamples, NULL);
      err += d*d;
    }
  }
  return err;
}

// The offload objective function
double _objFunc(unsigned int n,  const double * restrict x, 
		double * restrict grad, void * restrict my_func_data)
{
  double err;
  userData_t *uData = (userData_t *) my_func_data;

  // convert from double to float for speed
  for(int i=0; i < N_PARAM; i++) uData->param[i]=x[i];
  
  int nExamples = uData->nExamples;
  float *example = uData->example; // compiler workaround
  float *param = uData->param; // compiler workaround

#ifdef STAY_HERE
#pragma omp target map(to:param[0:N_PARAM]) map(alloc:example[0:nExamples])
  {
    err=0.; // initialize error here in case offload selected
    
    __assume_aligned(param, 64); __assume_aligned(example, 64);
#pragma omp parallel for simd reduction(+:err)
    for(int i=0; i < nExamples; i++) {
      float d=myFunc(i, param, example, nExamples, NULL);
      err += d*d;
    }
  }
#else
  return simdObjFunc(nExamples, example, param);
#endif

  return sqrt(err);
}
#pragma omp end declare target 

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
#pragma omp target
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
  float *example = uData->example;// compiler workaround
  float *param = uData->param;// compiler workaround
  //#pragma offload target(mic:MIC_DEV) in(example: length(0) FREE) in(param : length(0) FREE) {} 

  // free on the host
  if(uData->example) free(uData->example); uData->example=NULL;
  if(uData->param) free(uData->param); uData->param=NULL;
}

void offloadData(userData_t *uData)
{
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

  float *example = uData->example;
  float *param = uData->param;

#pragma omp declare target
#pragma omp target  map(alloc: example[0:nExamples], param[0:N_PARAM])
#pragma omp end declare target
  example[0]=0.; // deal with compiler bug not ending target declaration

  // read the data
  for(int exIndex=0; exIndex < uData->nExamples; exIndex++) {
    for(int i=0; i < nInput; i++) 
      fread(&uData->example[IN(i,uData->nExamples, exIndex)],1, sizeof(float), fn);
    for(int i=0; i < nOutput; i++) 
      fread(&uData->example[OUT(i,uData->nExamples, exIndex)],1, sizeof(float), fn);
  }
  
  // offload the data
  double startOffload=getTime();
  // Note: the in just allocates memory on the device
#pragma omp target update to(example[0:nExamples])

  uData->timeDataLoad = getTime() - startTime;

  if(fn!=stdin) fclose(fn);
}
