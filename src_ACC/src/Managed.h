#ifndef MANAGED_H
#define MANAGED_H

struct Managed
{
 public:
  void *operator new(size_t len) {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
    return ptr;
  }
  void operator delete(void *ptr, size_t len) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }
  void arrayAlloc(void **ptr, size_t len) {
    cudaMallocManaged(ptr, len);
    cudaDeviceSynchronize();
  }
  void arrayFree(void *ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }
};

#endif
