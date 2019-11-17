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


__host__ __device__ double schwefel(double*x,int dim){
  int    j;
  double s1 = 0;
  double s2 = 0;

  // T_{osz}
  transform_osz(x,dim);

  // T_{asy}^{0.2}
  transform_asy(x, 0.2, dim);

  for (j = 0; j < dim; j++) {
    s1 += x[j];
    s2 += (s1 * s1);
  }

  return(s2);
}

__host__ __device__ double F7(double *x, double *Ovector, double *anotherz, int *Pvector, double *r25, double *r50, double *r100)
{
     int    i;
    double result=0.0;
    //double anotherz[1000];
    double anotherz1[100];
    int s[7];
    s[0] = 50;
    s[1] = 25;
    s[2] = 25;
    s[3] = 100;
    s[4] = 50;
    s[5] = 25;
    s[6] = 25;

    double w[7];
    w[0] = 679.9025375867747;
    w[1] = 0.9321555273560842;
    w[2] = 2122.850158593588;
    w[3] = 0.5060110308419518;
    w[4] = 434.5961765462675;
    w[5] = 33389.62449652032;
    w[6] = 2.569238407592332;


    for(i = 0; i < N; i++) {
        anotherz[i] = x[i] - Ovector[i];
    }

    int* c = new int[1];
    c[0] = 0;
    for (i = 0; i < 7; i++)
    {
        rotateVector(i, c, anotherz1, s, anotherz, Pvector, r25, r50, r100);
        result += w[i] * schwefel(anotherz1, s[i]);
        //delete []anotherz1;
    }
    //double* z = new double[dimension-c];
    double z[700];
    //double* z = (double*) malloc(sizeof(double)*(N-c));
    //double* z = new double [N-c[0]];
    for (i = 300; i < N; i++)
        z[i-300] = anotherz[Pvector[i]];

    result += schwefel(z, 700);
    delete[] c;
    //return(result);
    return result;
}







