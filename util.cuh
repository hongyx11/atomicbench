#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#ifndef UTILCUH
#define UTILCUH


inline double gettime(void)
{
	struct timeval tp;
	gettimeofday( &tp, NULL );
	return tp.tv_sec + 1e-6 * tp.tv_usec;
}

inline void checkerr(cudaError_t e){
  if(e != cudaSuccess) printf("Error %s,  %s.\n", cudaGetErrorName(e), cudaGetErrorString(e));
}

inline void cudasetandsyncdevice(int device){
    checkerr(cudaSetDevice(device));
    cudaDeviceSynchronize();
}

template<typename T> inline
void cudanew(T **ptrref, size_t length){
  checkerr(cudaMalloc(ptrref, length * sizeof(T)));
  cudaMemset(*ptrref, 0, length * sizeof(T));
}

template<typename T> inline
void cudadelete(T * ptr){
  cudaFree(ptr);
}

template<typename T> inline
void cudamemcpy(T *dst, T * src, size_t len){
  cudaMemcpy(dst, src, len * sizeof(T), cudaMemcpyDefault);
}

#endif