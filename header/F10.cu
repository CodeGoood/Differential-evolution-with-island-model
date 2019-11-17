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

__host__ __device__ double F10(double *x, double *Ovector, double *anotherz, int *Pvector, double *r25, double *r50, double *r100)
{
     int    i;
    double result=0.0;
    //double anotherz[1000];
    double anotherz1[100];

    int s[20] = {50,50,25,25,100,100,25,25,50,25,100,25,100,50,25,25,25,100,50,25};

    double w[20];
    w[0] = 0.3127435060483861;
    w[1] = 15.12775129002843;
    w[2] = 2323.355051932917;
    w[3] = 0.0008059163159991055;
    w[4] = 11.42081047046656;
    w[5] = 3.554186402049341;
    w[6] = 29.98730853486525;
    w[7] = 0.99814770438828;
    w[8] = 1.615139127369899;
    w[9] = 1.512835249551407;
    w[10] = 0.6084813080072065;
    w[11] = 4464853.632306907;
    w[12] = 6.807672126970494e-05;
    w[13] = 0.1363174627449513;
    w[14] = 0.0007887145947891588;
    w[15] = 59885.12760160356;
    w[16] = 1.85232881407715;
    w[17] = 24.78342741548038;
    w[18] = 0.5431794716904831;
    w[19] = 39.24040541316312;

    for(i = 0; i < N; i++) {
        anotherz[i] = x[i] - Ovector[i];
    }

    int* c = new int[1];
    c[0] = 0;
    for (i = 0; i < 20; i++)
    {
        rotateVector(i, c, anotherz1, s, anotherz, Pvector, r25, r50, r100);
        result += w[i] * ackley(anotherz1, s[i]);
        //delete []anotherz1;
    }
    delete[] c;
//    return(result);
    return  result;
}


