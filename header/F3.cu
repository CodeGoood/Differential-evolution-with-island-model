#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <limits.h>
#include <math.h>
#include <cub/cub.cuh>
#include "para.cuh"
#include "F1.cuh"
#include "F2.cuh"

__host__ __device__ double ackley(double*x,int dim){
  double sum1 = 0.0;
  double sum2 = 0.0;
  double sum;
  int    i;

  // T_{osz}
  transform_osz(x,dim);
  
  // T_{asy}^{0.2}
  transform_asy(x, 0.2, dim);

  // lambda
  Lambda(x, 10, dim);

  for(i = dim - 1; i >= 0; i--) {
    sum1 += (x[i] * x[i]);
    sum2 += cos(2.0 * PI * x[i]);
  }

  sum = -20.0 * exp(-0.2 * sqrt(sum1 / dim)) - exp(sum2 / dim) + 20.0 + E;
  return(sum);
}

__host__ __device__ double F3(double *x, double *Ovector, double *anotherz)
{
    double result;
    int    i;
    //double Ovector[1000];
    //double anotherz[10000];
    //double *anotherz = new double [N];


    for(i = N - 1; i >= 0; i--) {
        anotherz[i] = x[i] - Ovector[i];
    }
    // T_{OSZ}
    // transform_osz(anotherz,dimension);

    //result = (anotherz,dimension);
    result = ackley(anotherz,N);
    //return(result);
    //if(result==0)
//        for(int m = 0;m<1000;m++)
  //          printf("vv:%lf ",x[m]);


    return result;
}






