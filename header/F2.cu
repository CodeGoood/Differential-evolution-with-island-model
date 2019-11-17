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

__host__ __device__ void transform_asy(double* z, double beta, int dim)
{
  for (int i = 0; i < dim; ++i)
    {
      if (z[i]>0)
        {
          z[i] = pow(z[i], 1 + beta * i/((double) (dim-1)) * sqrt(z[i]) );
        }
    }
}

__host__ __device__ void Lambda(double* z, double alpha, int dim)
{
  for (int i = 0; i < dim; ++i)
    {
      z[i] = z[i] * pow(alpha, 0.5 * i/((double) (dim-1)) );
    }
}

__host__ __device__ double rastrigin(double*x,int dim){
  double sum = 0;
  int    i;
  
  // T_{osz}
  transform_osz(x, dim);
  
  // T_{asy}^{0.2}
  transform_asy(x, 0.2, dim);

  // lambda
  Lambda(x, 10, dim);

  for(i = dim - 1; i >= 0; i--) {
    sum += x[i] * x[i] - 10.0 * cos(2 * PI * x[i]) + 10.0;
  }
  
  return(sum);
}

__host__ __device__ double F2(double *x, double *Ovector, double *anotherz)
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
    result = rastrigin(anotherz,N);
    //return(result);
    //sub_fit[0] = result;
    return result;
}



