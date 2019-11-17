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
#include "F4.cuh"

__host__ __device__ double F5(double *x, double *Ovector, double *anotherz, int *Pvector, double *r25, double *r50, double *r100)
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
    w[0] = 0.1807559981875739;
    w[1] = 9081.137957372908;
    w[2] = 24.27180711217444;
    w[3] = 1.86308808032975e-06;
    w[4] = 17698.08075760589;
    w[5] = 0.0002815181918094626;
    w[6] = 0.01525403796219806;

    for(i = 0; i < N; i++) {
        anotherz[i] = x[i] - Ovector[i];
    }

    int* c = new int[1];
    c[0] = 0;
    for (i = 0; i < 7; i++)
    {
        rotateVector(i, c, anotherz1, s, anotherz, Pvector, r25, r50, r100);
        result += w[i] * rastrigin(anotherz1, s[i]);
        //delete []anotherz1;
    }
    //double* z = new double[dimension-c];
    double z[700];
    //double* z = (double*) malloc(sizeof(double)*(N-c));
    //double* z = new double [N-c[0]];
    for (i = 300; i < N; i++)
        z[i-300] = anotherz[Pvector[i]];

    result += rastrigin(z, 700);
    delete[] c;
    //return(result);
    return result;
}

