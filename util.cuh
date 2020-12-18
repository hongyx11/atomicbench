#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <time.h>
#include <sys/time.h>

#ifndef UTILCUH
#define UTILCUH

#ifdef USE_FLOAT
#define real float
#else
#define real double
#endif 

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


__global__ void generaternd(double *oarr){
  unsigned long int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  curandState state;
  curand_init((unsigned long long)clock() + tId, 0, 0, &state);
  oarr[tId] = curand_uniform_double(&state);
}

__global__ void generaternd(float *oarr){
  unsigned long int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  curandState state;
  curand_init((unsigned long long)clock() + tId, 0, 0, &state);
  oarr[tId] = curand_uniform(&state);
}


inline void initrandomdata(real *hptr, size_t n){
  unsigned int bx = 128;
  unsigned int gx = n / bx + (n%bx != 0);
  real *dptr_rnd; cudanew(&dptr_rnd, n);
  generaternd<<<gx,bx>>>(dptr_rnd);
  checkerr(cudaGetLastError());
  cudamemcpy(hptr, dptr_rnd, n);
  cudadelete(dptr_rnd);
}




/**
* CPU and GPU reduce sum implementation
* 
*/
inline real cpureduce(real *arr, size_t len){
  real res = 0.0;
  #pragma omp parallel for reduction(+:res)
  for(size_t i=0; i < len; i++){
    res += arr[i];
  }
  return res;
}


// atomic kernel 
inline __global__ void atomickernel(real *arr, real *outarr){
  unsigned long int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  atomicAdd(&outarr[tId % 128], arr[tId]);
}



// reduce kernel

template <unsigned int blockSize, typename T>
__device__ void warpReduce(volatile T *sdata, unsigned int tid){
  if(blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if(blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if(blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if(blockSize >= 8) sdata[tid] += sdata[tid + 4];
  if(blockSize >= 4) sdata[tid] += sdata[tid + 2];
  if(blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template<unsigned int blockSize, typename T>
__global__ void reduce6(T *g_idata, T *g_odata, unsigned int n){
  extern __shared__ T sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + tid;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  sdata[tid] = 0.0;
  while(i < n){
    sdata[tid] += g_idata[i] + g_idata[i+blockSize];
    i += gridSize;
  }
  __syncthreads();
  if(blockSize >= 512){if(tid < 256){sdata[tid] += sdata[tid+256];} __syncthreads();}
  if(blockSize >= 256){if(tid < 128){sdata[tid] += sdata[tid+128];} __syncthreads();}
  if(blockSize >= 128){if(tid < 64){sdata[tid] += sdata[tid+64];} __syncthreads();}
  if(tid < 32) warpReduce<blockSize,T>(sdata, tid);
  if(tid == 0) g_odata[blockIdx.x] = sdata[0];
}



#endif