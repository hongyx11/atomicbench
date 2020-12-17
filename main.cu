#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define XSTR(x) STR(x)
#define STR(x) #x
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
#endif

__global__
void generaternd(double*oarr){
  unsigned long int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  curandState state;
  curand_init((unsigned long long)clock() + tId, 0, 0, &state);
  oarr[tId] = curand_uniform_double(&state);
}

__global__
void hellothd(double *arr, double *outarr){
  unsigned long int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  double val[25];
  double tmpval = 0.0;
  for(int i=0; i<50; i++){
    tmpval += 0.1 * 50 * arr[tId] / 84.; 
  }
  for(int i=0; i<40; i++){
    atomicAdd(&outarr[i*128 + tId % 128], val[i]);
  }
}

void checkerr(cudaError_t e){
  if(e != cudaSuccess) printf("Error %s,  %s.\n", cudaGetErrorName(e), cudaGetErrorString(e));
}

inline 
void cudasetandsyncdevice(int device){
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

double gettime(void)
{
	struct timeval tp;
	gettimeofday( &tp, NULL );
	return tp.tv_sec + 1e-6 * tp.tv_usec;
}

int main(int argc, char *argv[]){
  size_t n = atoi(argv[1]);
  printf("number size %d cost mem %.5f GB \n", n, n*8/1000./1000./1000.);
  double * hptr = new double[n];
  double * hout = new double[128];
  // for(int j=0; j<n; j++) hptr[j] = (double)rand() / (double)RAND_MAX;
  unsigned int bx = 128;
  unsigned int gx = n / bx + (n%bx != 0);
  double *dptr_rnd; cudanew(&dptr_rnd, n);
  generaternd<<<gx,bx>>>(dptr_rnd);
  checkerr(cudaGetLastError());
  cudamemcpy(hptr, dptr_rnd, n);
  cudadelete(dptr_rnd);
  double cpusum = 0.0;
  printf("cpu sum is %f \n", cpusum);
  double * dptr, *outarr;
  cudanew(&dptr, n);
  cudanew(&outarr, 128 * 50);
  cudamemcpy(dptr, hptr, n);
  cudasetandsyncdevice(0);
  double t1 = gettime();
  hellothd<<<gx,bx>>>(dptr,outarr);
  cudasetandsyncdevice(0);
  double t2 = gettime();
  printf("kernel time is %.6f s \n", t2 - t1);
  cudamemcpy(hout, outarr, 128);
  double gpusum = 0.0;
  // for(int j=0; j<128;j++) gpusum += hout[j];
  // printf("gpu sum is %f , diff with cpu is %.16f \n", gpusum, (gpusum - cpusum)/n );
  cudadelete(dptr);
  cudadelete(outarr);
  delete[] hptr;
  delete[] hout;
}