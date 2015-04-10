// Rob Farber

// Define key GPU characteristics
#define NUM_SMX 15
#define NUM_ACTIVE_SMX_QUEUE 16
#define VEC_LEN 32

#define restrict

// helper macros to index into the example array
#define IN(i,nExamples,j)  (i*nExamples+j)
#define OUT(i,nExamples,j)  ((i+N_INPUT)*nExamples+j)

// Define the Sigmoid
#ifdef USE_LINEAR
#define G_DESC_STRING "generated_PCA_func LINEAR()"
__host__ __device__
inline float G(float x) { return( x ) ;} 
#define G_ESTIMATE 0 
#elif USE_TANH
#define G_DESC_STRING "generated_func tanh()"
__host__ __device__
inline float G(float x) { return( tanhf(x) ) ;} 
#define G_ESTIMATE 7 // estimate 7 flops for G
#elif LOGISTIC
#define G_DESC_STRING "generated func logistic()"
__host__ __device__
inline float G(float x) { return( 1.f/(1.f+expf(-x)) ) ;} 
#define G_ESTIMATE 7 // estimate flops for G
#else // Use Elliott function
#define G_DESC_STRING "generated func Eliott activation: x/(1+fabsf(x))"
__host__ __device__
inline float G(float x) { return( x/(1.f+fabsf(x)) ) ;} 
#define G_ESTIMATE 3 // estimate flops for G
#endif
