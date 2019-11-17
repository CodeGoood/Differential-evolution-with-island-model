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


__host__ __device__ double F11(double *x, double *Ovector, double *anotherz, int *Pvector, double *r25, double *r50, double *r100)
{
     int    i;
    double result=0.0;
    //double anotherz[1000];
    double anotherz1[100];

    int s[20] = {50,50,25,25,100,100,25,25,50,25,100,25,100,50,25,25,25,100,50,25};

    double w[20];
    w[0] = 0.01613441050029006;
    w[1] = 0.1286113102576636;
    w[2] = 0.001204762698494153;
    w[3] = 0.3492354534770768;
    w[4] = 3.988703102369622;
    w[5] = 7.446972342274892;
    w[6] = 2.613875947071485;
    w[7] = 1.860151050804902e-05;
    w[8] = 0.07799241383970082;
    w[9] = 4946500.039233495;
    w[10] = 907.5677350872909;
    w[11] = 1245.438955040323;
    w[12] = 0.0001277872704005029;
    w[13] = 0.002545171687133515;
    w[14] = 0.01229630267562622;
    w[15] = 0.2253262515782924;
    w[16] = 16011.68013995448;
    w[17] = 4.152882294482328;
    w[18] = 4208.608600430911;
    w[19] = 8.983034451382988e-06;


    for(i = 0; i < N; i++) {
        anotherz[i] = x[i] - Ovector[i];
    }

    int* c = new int[1];
    c[0] = 0;
    for (i = 0; i < 20; i++)
    {
        rotateVector(i, c, anotherz1, s, anotherz, Pvector, r25, r50, r100);
        result += w[i] * schwefel(anotherz1, s[i]);
        //delete []anotherz1;
    }
    delete[] c;
//    return(result);
    return result;
}

