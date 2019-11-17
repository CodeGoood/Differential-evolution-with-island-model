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

#include <sstream>
#include <vector>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include "para.cuh"
#include "F1.cuh"
#include "F2.cuh"
#include "F3.cuh"
#include "F4.cuh"
#include "F7.cuh"

__host__ __device__ double rosenbrock(double*x,int dim){
  int    j;
  double oz,t;
  double s = 0.0;
  j = dim - 1;

  for (--j; j >= 0; j--) {
    oz = x[j + 1];
    t  = ((x[j] * x[j]) - oz);
    s += (100.0 * t * t);
    t  = (x[j] - 1.0);
    s += (t * t);
  }
  return(s);
}

__host__ __device__ double F12(double *x, double *Ovector, double *anotherz, int *Pvector, double *r25, double *r50, double *r100)
{
     int    i;
    double result=0.0;

    for(i = 0; i < N; i++) {
        anotherz[i] = x[i] - Ovector[i];
    }
    
    result = rosenbrock(anotherz, N);
    //return(result);
    return  result;
}






