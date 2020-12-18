#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "util.cuh"

#ifdef USE_FLOAT
#define real float
#else
#define real double
#endif 

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


void initrandomdata(real *hptr, size_t n){
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
real cpureduce(real *arr, size_t len){
  real res = 0.0;
  #pragma omp parallel for reduction(+:res)
  for(size_t i=0; i < len; i++){
    res += arr[i];
  }
  return res;
}


__global__ void atomickernel(real *arr, real *outarr){
unsigned long int tId = threadIdx.x + (blockIdx.x * blockDim.x);
atomicAdd(&outarr[tId % 128], arr[tId]);
// double val[25];
// double tmpval = 0.0;
// for(int i=0; i<50; i++){
//   tmpval += 0.1 * 50 * arr[tId] / 84.; 
// }
// for(int i=0; i<40; i++){
//   atomicAdd(&outarr[i*128 + tId % 128], val[i]);
// }
}





int main(int argc, char *argv[]){
  printf ("\n This benchmark computes reduce operation using atomic kernel / customized kernel, \n"
  " on NVIDIA P100 / V100 / A100. we benchmark on scalar. \n\n");
  size_t n = atoi(argv[1]);
  printf(" Number of problem size %lu whole problem szie takes memory %.5f GB \n", n, n * 8./1000./1000./1000.);
  real * hiarr = new real[n];
  // init rnd data
  initrandomdata(hiarr, n);
  real cpusum = cpureduce(hiarr, n);
  printf(" cpu sum is %f \n", cpusum);
  // prepare gpu memory
  real * diarr, *doarr; // gpu ptr
  real * hout = new real[128]; // cpu ptr
  size_t redcueslot = 128;
  size_t paramsize = 1;
  cudanew(&diarr, n); cudanew(&doarr, redcueslot * paramsize);
  cudamemcpy(diarr, hiarr, n);
  // prepare gpu kernel size
  size_t bx = 128;
  size_t gx = n / bx + (n%bx != 0);
  cudasetandsyncdevice(0);
  real t1 = gettime();
  #ifdef USE_ATOMIC
  printf("\n we are using atomic operation \n");
  atomickernel<<<gx,bx>>>(diarr,doarr);
  #endif
  // get results back to cpu
  cudasetandsyncdevice(0);
  real t2 = gettime();
  printf(" kernel time is %.6f ms \n", (t2 - t1)*1000. );
  cudamemcpy(hout, doarr, 128);
  real gpusum = 0.0;
  for(int j=0; j<128;j++) gpusum += hout[j];
  printf(" gpu sum is %f , diff with cpu is %e \n", gpusum, abs((gpusum - cpusum))/n );
  cudadelete(diarr);
  cudadelete(doarr);
  delete[] hiarr;
  delete[] hout;
}