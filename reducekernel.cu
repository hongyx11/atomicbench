#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "util.cuh"


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
// prepare gpu kernel size
  const size_t bx = 128;
  size_t gx = n / bx / 4 + (n%bx != 0);
  // prepare gpu memory
  real * diarr, *doarr; // gpu ptr
  real * hout = new real[gx]; // cpu ptr
  size_t redcueslot = gx;
  size_t paramsize = 1;
  cudanew(&diarr, n); cudanew(&doarr, redcueslot * paramsize);
  cudamemcpy(diarr, hiarr, n);
  printf("\n we are using reduce operation \n");
  cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start);
  cudaDeviceSynchronize();
  reduce6<128,real><<<gx,128,128*sizeof(real)>>>(diarr,doarr,n);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  showstat(milliseconds * 1e-3, n);
  cudamemcpy(hout, doarr, redcueslot);
  real gpusum = 0.0;
  for(int j=0; j<redcueslot;j++) gpusum += hout[j];
  printf(" gpu sum is %f , diff with cpu is %e \n", gpusum, abs((gpusum - cpusum))/n );
  cudadelete(diarr);
  cudadelete(doarr);
  delete[] hiarr;
  delete[] hout;
}