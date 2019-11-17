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

__host__ __device__ int sign(double x)
{
  if (x > 0) return 1;
  if (x < 0) return -1;
  return 0;
}

__host__ __device__ double hat(double x)
{
  if (x==0)
    {
      return 0;
    }
  else
    {
      return log(abs(x));
    }
}

__host__ __device__ double c1(double x)
{
  if (x>0)
    {
      return 10;
    }
  else
    {
      return 5.5;
    }
}

__host__ __device__ double c2(double x)
{
  if (x>0)
    {
      return 7.9;
    }
  else
    {
      return 3.1;
    }
}

__host__ __device__ void transform_osz(double* z, int dim)
{
  // apply osz transformation to z
  for (int i = 0; i < dim; ++i)
    {
      z[i] = sign(z[i]) * exp( hat(z[i]) + 0.049 * ( sin( c1(z[i]) * hat(z[i]) ) + sin( c2(z[i])* hat(z[i]) )  ) ) ;
    }
}

__host__ __device__ double elliptic(double*x,int dim) {
  double result = 0.0;
  int    i;
  
  transform_osz(x, dim);

  // for(i = dim - 1; i >= 0; i--) {
  for(i=0; i<dim; i++)
    {
      // printf("%f\n", pow(1.0e6,  i/((double)(dim - 1)) ));
      result += pow(1.0e6,  i/((double)(dim - 1)) ) * x[i] * x[i];
    }
  
  return(result);
}

__host__ __device__ double F1(double *x, double *Ovector, double *anotherz)
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

    // result = elliptic(anotherz,dimension);
    result = elliptic(anotherz, N);
    //sub_fit[0] = result;
    return result;
}
